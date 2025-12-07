#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include <numeric>

namespace mlir {
namespace tutorial {

#define GEN_PASS_DEF_AFFINEFULLUNROLL
#include "lib/Transform/Affine/Passes.h.inc"

using mlir::affine::AffineForOp;

// A pass that manually walks the IR
struct AffineFullUnroll : impl::AffineFullUnrollBase<AffineFullUnroll> {
  using AffineFullUnrollBase::AffineFullUnrollBase;

  void runOnOperation() {
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
