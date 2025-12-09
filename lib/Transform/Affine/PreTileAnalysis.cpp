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
#include <set>
#include <cstdlib>
#include <fstream>

namespace mlir {
namespace polyTiling {

#define GEN_PASS_DEF_PRETILEANALYSIS
#include "lib/Transform/Affine/Passes.h.inc"

using mlir::affine::AffineForOp;
using mlir::affine::AffineLoadOp;
using mlir::affine::AffineStoreOp;
using mlir::ValueRange;


/// Extract coefficient of IV `ivIndex` in affine expression `expr`.
static float getCoeff(AffineExpr expr, Value ivIndex, ValueRange mapOperands) {
  if (auto dim = expr.dyn_cast<AffineDimExpr>()) {
    unsigned pos = dim.getPosition();
    Value operand = mapOperands[pos];
    if (operand == ivIndex) return 1;
    return 0;
  }
  if (auto c = expr.dyn_cast<AffineConstantExpr>()) {
    return 0;
  }
  if (auto sym = expr.dyn_cast<AffineSymbolExpr>()) {
    return 0;
  }
  if (auto bin = expr.dyn_cast<AffineBinaryOpExpr>()) {
    int64_t l = getCoeff(bin.getLHS(), ivIndex, mapOperands);
    int64_t r = getCoeff(bin.getRHS(), ivIndex, mapOperands);
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
  SmallVector<int64_t, 8> tileLBs;
  SmallVector<int64_t, 8> tileUBs;
  SmallVector<int64_t, 8> strides;
  for (AffineForOp f : loopBand) {
    ivs.push_back(f.getInductionVar());
    tileLBs.push_back(f.getConstantLowerBound());
    tileUBs.push_back(f.getConstantUpperBound());
    strides.push_back(f.getStep());
  }
  unsigned n = ivs.size();


  // Raw reuse accumulator (one entry per loop in band)
  std::vector<double> raw(n, 0.0);

  // Walk operations in the innermost loop (conservative: everything in its region)
  AffineForOp innermost = loopBand.back();

  // maintain a set of the memref values that have been seen
  //std::set<Value> memrefValues;
  //std::vector<std::vector<double>> memoryFootprintPolynomialTerms; // M x N matrix, N = number of loop IVs.
  // [[a1, a2, a3...], [b1, b2, b3...]...] ==> footprint = (a1 * t1 * a2 * t2 * a3 * t3 ... * aN * tN) + (b1 * t1 * b2 * t2 * b3 * t3 ... * bN * tN) + ...  
  
  // print innermost size

  std::vector<std::vector<std::vector<float>>> memoryFootprintPolynomialTerms;

  std::vector<std::pair<Value, AffineMap>> uniqueMemRefs;

  innermost.walk([&](Operation *op) {
    // Only process affine.load / affine.store
    if (!isa<affine::AffineLoadOp>(op) &&
        !isa<affine::AffineStoreOp>(op)) {
//          llvm::errs() << "  op: " << op->getName() << "\n";
          return WalkResult::advance();
    }

    // print that we have an affine load or store
  
    // Must declare loadOp / storeOp BEFORE using them
    auto loadOp  = dyn_cast<affine::AffineLoadOp>(op);
    auto storeOp = dyn_cast<affine::AffineStoreOp>(op);
    bool isStore = (bool)storeOp;
  
    // Polygeist MLIR has NO MemoryOpInterface → use direct getMemRefType()
    MemRefType mt = loadOp
        ? loadOp.getMemRefType()
        : storeOp.getMemRefType();
  
  
    // Access map
    AffineMap map = loadOp
        ? loadOp.getAffineMap()
        : storeOp.getAffineMap();
    
    Value memRef = loadOp ? loadOp.getOperand(loadOp.getMemRefOperandIndex()) : storeOp.getOperand(storeOp.getMemRefOperandIndex());

    unsigned numIdx = map.getNumResults();
    ValueRange mapOperands = ValueRange();
    AffineLoadOp loadOpCast;
    AffineStoreOp storeOpCast;
    if ((bool) loadOp) {
      // autocast Operation to AffineLoadOp
      loadOpCast = dyn_cast<affine::AffineLoadOp>(op);
      mapOperands = loadOpCast.getMapOperands();
    } else {
      // autocast Operation to AffineStoreOp
      storeOpCast = dyn_cast<affine::AffineStoreOp>(op);
      mapOperands = storeOpCast.getMapOperands();
    }

    int64_t num_reuses = 1;
    bool isReusing[n];

    for (unsigned k = 0; k < n; ++k) {
      isReusing[k] = true;
    }

    float termsForOp[numIdx][n];
    int64_t numIterPts[numIdx];
    // Initialize numIterPts to 0 (will accumulate sum, or set to 1 if no dependency)
    for (unsigned d = 0; d < numIdx; ++d) {
      numIterPts[d] = 0;
    }

    // For each loop IV in the band

    // terms for polynomial: (sum of |coeff| * tile size for each index) * (...) for each expression, summed up
    // num_iter_pts[d] = sum over k ceiling[abs(ub[k] - lb[k] + 1) / (abs(coeff) * strides[k])] if coeff > 0 else 1
    // reuseFactor = product over num_iter_pts.
    
    for (unsigned k = 0; k < n; ++k) {
      for (unsigned d = 0; d < numIdx; ++d) {
        AffineExpr expr = map.getResult(d);
        float coeff = getCoeff(expr, ivs[k], mapOperands);
        
        termsForOp[d][k] = std::abs(coeff);
        if (std::abs(coeff) > 0 && strides[k] > 0) {
           isReusing[k] = false;
           numIterPts[d] += (int64_t) std::ceil(std::abs(tileUBs[k] - tileLBs[k]) / ((float) strides[k]));
        }
      }
    }

    // If numIterPts[d] is still 0 (no dependency on any loop), set to 1
    for (unsigned d = 0; d < numIdx; ++d) {
      if (numIterPts[d] == 0) {
        numIterPts[d] = 1;
      }
    }

    int64_t reuseFactor = 1;
    for (unsigned d = 0; d < numIdx; ++d) {
      reuseFactor *= numIterPts[d];
    }

    for (unsigned k = 0; k < n; ++k) {
      if (isReusing[k]) {
        raw[k] += reuseFactor;
      }
    }

    bool foundMemRef = false;
    for (unsigned i = 0; i < uniqueMemRefs.size(); ++i) {
      if (uniqueMemRefs[i].first == memRef && uniqueMemRefs[i].second == map) {
        foundMemRef = true;
        break;
      }
    }

    if (!foundMemRef) {
      uniqueMemRefs.push_back(std::make_pair(memRef, map));
      std::vector<std::vector<float>> newTerms(numIdx, std::vector<float>(n, 0.0));
      for (unsigned d = 0; d < numIdx; ++d) {
        for (unsigned k = 0; k < n; ++k) {
          newTerms[d][k] = termsForOp[d][k];
        }
      }
      memoryFootprintPolynomialTerms.push_back(newTerms);
    }
    
    return WalkResult::advance();
  }); // end walk

  // Normalize raw -> gamma (largest dimension becomes 1.0)
  double maxv = 0.0;
  for (double v : raw) if (v > maxv) maxv = v;
  if (maxv <= 0.0) maxv = 1.0; // avoid divide by zero, fallback equalization if desired


  // get environment variable on what txt file to write to
  const char* outputFileEnv = std::getenv("OUTPUT_FILE");
  if (!outputFileEnv) {
    llvm::errs() << "Error: OUTPUT_FILE environment variable not set\n";
    return;
  }
  std::string outputFile = outputFileEnv;
  std::ofstream outputFileStream(outputFile);

  // write out the raw values on top line 
  for (unsigned i = 0; i < n; ++i) {
    outputFileStream << raw[i] / maxv << " ";
  }
  outputFileStream << "\n";
  
  // write out the memory footprint polynomial terms, using a newline terminator after each j, and two newlines after each i
  for (unsigned i = 0; i < memoryFootprintPolynomialTerms.size(); ++i) {
    for (unsigned j = 0; j < memoryFootprintPolynomialTerms[i].size(); ++j) {
      for (unsigned k = 0; k < n; ++k) {
        outputFileStream << memoryFootprintPolynomialTerms[i][j][k] << " ";
      }
      outputFileStream << "\n";
    }
    outputFileStream << "\n\n";
  }

  // close the output file
  outputFileStream.close();
}

// A pass that manually walks the IR
struct PreTileAnalysis : impl::PreTileAnalysisBase<PreTileAnalysis> {
  using PreTileAnalysisBase::PreTileAnalysisBase;

  void runOnOperation() {
    int nestId = 0;
    int loop_count = 0;
    getOperation()->walk([&](AffineForOp op) {

      //printLoopBounds(op, loop_count++);
      
      if (op->getParentOfType<AffineForOp>()) {
      return WalkResult::advance();  
      }
    
      SmallVector<AffineForOp> loopBand;
      getPerfectlyNestedLoops(loopBand, op);
      if (loopBand.size() >= 2) {
        llvm::errs() << "Nest #" << ++nestId << ":\n";
        //analyzeLoopPermutability(loopBand);

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



} // namespace polyTiling
} // namespace mlir
