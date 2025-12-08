#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include <numeric>


// includes for the reuse shit
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "llvm/Support/raw_ostream.h"
#include <vector>
#include <algorithm>

#include "mlir/Pass/PassManager.h"       // actually correct
#include "mlir/Dialect/Affine/Passes.h"  // for createAffineLoopConvertPass
#include "lib/Transform/Affine/Passes.h.inc"
#include "lib/Transform/Affine/RaiseToAffine.h"


namespace mlir {
namespace tutorial {

#define GEN_PASS_DEF_AFFINEFULLUNROLL
#include "lib/Transform/Affine/Passes.h.inc"

using mlir::affine::AffineForOp;


/// Extract coefficient of IV `ivIndex` in affine expression `expr`.
static int64_t getCoeff(AffineExpr expr, unsigned ivIndex) {
  if (auto dim = expr.dyn_cast<AffineDimExpr>()) {
    if (dim.getPosition() == ivIndex) return 1;
    return 0;
  }
  if (auto c = expr.dyn_cast<AffineConstantExpr>()) {
    return 0;
  }
  if (auto sym = expr.dyn_cast<AffineSymbolExpr>()) {
    return 0;
  }
  if (auto bin = expr.dyn_cast<AffineBinaryOpExpr>()) {
    int64_t l = getCoeff(bin.getLHS(), ivIndex);
    int64_t r = getCoeff(bin.getRHS(), ivIndex);
    if (bin.getKind() == AffineExprKind::Add)
      return l + r;
    if (bin.getKind() == AffineExprKind::Mul) {
      // one side must be constant for affine
      if (auto cst = bin.getLHS().dyn_cast<AffineConstantExpr>())
        return cst.getValue() * r;
      if (auto cst = bin.getRHS().dyn_cast<AffineConstantExpr>())
        return cst.getValue() * l;
    }
  }
  return 0;
}


/// Compute row-major strides for memref type.
static std::vector<int64_t> getStrides(MemRefType mt) {
  int rank = mt.getRank();
  std::vector<int64_t> s(rank, 1);
  for (int i = rank - 2; i >= 0; --i) { // NOTE: this assumes static dims
    s[i] = s[i + 1] * mt.getDimSize(i + 1);
  }
  return s;
}

/// Analyze a loop band (outer->inner in 'loopBand') and compute raw reuse
/// and normalized gamma per loop in the band. Prints results via errs().
static void analyzeAffineBand(ArrayRef<AffineForOp> loopBand) {
  if (loopBand.empty()) return;


  // The band is expected outer->inner. Collect IV order accordingly.
  SmallVector<Value, 8> ivs;
  for (AffineForOp f : loopBand)
    ivs.push_back(f.getInductionVar());
  unsigned n = ivs.size();


  // Raw reuse accumulator (one entry per loop in band)
  std::vector<double> raw(n, 0.0);

  // Walk operations in the innermost loop (conservative: everything in its region)
  AffineForOp innermost = loopBand.back();

  // print innermost size

  innermost.walk([&](Operation *op) {
    // Only process affine.load / affine.store
    if (!isa<affine::AffineLoadOp>(op) &&
        !isa<affine::AffineStoreOp>(op)) {
          llvm::errs() << "  op: " << op->getName() << "\n";
          return;
        }

    // print that we have an affine load or store
    llvm::errs() << "  Affine load or store\n";

  
    // Must declare loadOp / storeOp BEFORE using them
    auto loadOp  = dyn_cast<affine::AffineLoadOp>(op);
    auto storeOp = dyn_cast<affine::AffineStoreOp>(op);
    bool isStore = (bool)storeOp;
  
    // Polygeist MLIR has NO MemoryOpInterface → use direct getMemRefType()
    MemRefType mt = loadOp
        ? loadOp.getMemRefType()
        : storeOp.getMemRefType();
  
    auto strides = getStrides(mt);
  
    // Access map
    AffineMap map = loadOp
        ? loadOp.getAffineMap()
        : storeOp.getAffineMap();
    
    Value memRef = loadOp ? loadOp.getOperand(loadOp.getMemRefOperandIndex()) : storeOp.getOperand(storeOp.getMemRefOperandIndex());
    llvm::errs() << "  MemRef: " << memRef << "\n";
    unsigned numIdx = map.getNumResults();

  
    // For each loop IV in the band
    for (unsigned k = 0; k < n; ++k) {
      int64_t Lk = 0;
      for (unsigned d = 0; d < numIdx; ++d) {
        AffineExpr expr = map.getResult(d);
        int64_t coeff = getCoeff(expr, k);
        Lk += coeff * strides[d];
      }
  
      if (Lk == 0)
        raw[k] += 1.0;          // temporal reuse
      else if (std::abs(Lk) == 1)
        raw[k] += 1.0;          // spatial stride-1 reuse
  
      if (isStore)
        raw[k] += 1.0;          // store bonus
    }
  }); // end walk

  // Normalize raw -> gamma (largest dimension becomes 1.0)
  double maxv = 0.0;
  for (double v : raw) if (v > maxv) maxv = v;
  if (maxv <= 0.0) maxv = 1.0; // avoid divide by zero, fallback equalization if desired

  llvm::errs() << "  Affine band reuse (depth=" << n << ")\n";
  for (unsigned i = 0; i < n; ++i) {
    double gamma = raw[i] / maxv;
    llvm::errs() << "    loop " << i << " raw=" << raw[i] << " gamma=" << gamma << "\n";
  }
  llvm::errs() << "----------------------------------------\n";
}


// A pass that manually walks the IR
struct AffineFullUnroll : impl::AffineFullUnrollBase<AffineFullUnroll> {
  using AffineFullUnrollBase::AffineFullUnrollBase;

  void runOnOperation() {

    auto func = getOperation();

    {
      mlir::PassManager pm(func->getContext());
      pm.addPass(createRaiseToAffine());
      if (failed(pm.run(func))) {
        signalPassFailure();
        return;
      }
    }

    int nestId = 0;
    int loop_count = 0;
    getOperation()->walk([&](AffineForOp op) {

      printLoopBounds(op, loop_count++);
      
      if (op->getParentOfType<AffineForOp>()) {
      return WalkResult::advance();  
      }
    
      SmallVector<AffineForOp> loopBand;
      getPerfectlyNestedLoops(loopBand, op);
      if (loopBand.size() >= 2) {
        llvm::errs() << "Nest #" << ++nestId << ":\n";
        analyzeLoopPermutability(loopBand);

        // <<-- NEW: compute affine reuse for this band
        // We inserted analyzeAffineBand here so it runs for each discovered band.
        analyzeAffineBand(loopBand);
        // -->> end new
      }
    
      return WalkResult::advance();
    });
  }

  void printLoopBounds(AffineForOp loop, int id) {
    llvm::errs() << "Loop #" << id << ":\n";  
    // Bounds
    auto lbConst = loop.getConstantLowerBound();
    auto ubConst = loop.getConstantUpperBound();
    auto step = loop.getStep();
    
    llvm::errs() << "  Bounds: ";
    if (lbConst && ubConst) {
      llvm::errs() << "[" << lbConst << ", " << ubConst << ") ";
    } else {
      llvm::errs() << "LB=" << loop.getLowerBoundMap();
      llvm::errs() << " UB=" << loop.getUpperBoundMap();
    }
    llvm::errs() << "step=" << step << "\n";
    
    // Iterations (if constant bounds)
    if (lbConst && ubConst) {
      int64_t iterations = (ubConst - lbConst + step - 1) / step;
      llvm::errs() << "  Iterations: " << iterations << "\n";
    }
    
    // Induction variable
    llvm::errs() << "  IV: " << loop.getInductionVar() << "\n";
    
    // Nesting level (heuristic: count parent loops)
    int nestingLevel = 0;
    auto *parentOp = loop->getParentOp();
    while (isa<AffineForOp>(parentOp)) {
      nestingLevel++;
      parentOp = parentOp->getParentOp();
    }
    llvm::errs() << "  Nesting: level " << nestingLevel << "\n";
    
    // Body size (operations)
    int bodyOps = 0;
    loop.getBody()->walk([&](Operation*) { bodyOps++; return WalkResult::advance(); });
    llvm::errs() << "  Body ops: " << bodyOps << "\n";
    
    llvm::errs() << "----------------------------------------\n";
  }

  void analyzeLoopPermutability(ArrayRef<AffineForOp> loopBand) {
    size_t nLoops = loopBand.size();
    
    llvm::errs() << "  Nest depth: " << nLoops << "\n";
    
    // Print loop identifiers
    llvm::errs() << "  Loops:\n";

    // Test ALL pairwise interchanges (i <-> j)
    llvm::errs() << "\n  Permutability (pairwise swaps):\n";
    for (size_t i = 0; i < nLoops; ++i) {
      for (size_t j = i + 1; j < nLoops; ++j) {
        // Create permutation: swap i and j
        SmallVector<unsigned> perm(nLoops);
        std::iota(perm.begin(), perm.end(), 0u);  // [0,1,2,3,...]
        std::swap(perm[i], perm[j]);              // Swap levels i,j

        // Test legality
        bool result = affine::isValidLoopInterchangePermutation(loopBand, perm);

        llvm::errs() << "    L" << i << " <-> L" << j << ": " 
                     << (result ? "LEGAL" : "ILLEGAL") << "\n";
      }
    }

    llvm::errs() << "\n";
  }
};



} // namespace tutorial
} // namespace mlir
