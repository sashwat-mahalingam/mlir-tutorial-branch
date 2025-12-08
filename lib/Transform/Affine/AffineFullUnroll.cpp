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
    // Hard coding number of multi-cache levels to 3 for now 
    int numLevels = 3;
    // Hard code cache sizes for L1, L2, L3 cache
    std::vector<uint64_t> cacheBytes = {
                  8ULL*1024*1024,    // L3: 8MB  
                  1ULL*1024*1024,    // L2: 1MB
                  128ULL*1024};      // L1: 128KB
    uint64_t elemBytes = 4; 

    std::vector<double> cacheElems(numLevels);
    for (size_t lvl = 0; lvl < numLevels; ++lvl) {
      cacheElems[lvl] = (double)cacheBytes[lvl] / (double)elemBytes;
    }

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

      
      // tile size selection
      SmallVector<AffineForOp, 8> origBand;
      getPerfectlyNestedLoops(origBand, op);
      
      if (origBand.size() < 2) return;  // Need at least 2D for tiling
      
      size_t n = origBand.size();
      
      // Compute dimensional reuse γ once for this n-D band
      SmallVector<double, 8> gamma = computeDimensionalReuse(origBand);
      
      // Current innermost band starts as original band
      SmallVector<AffineForOp, 8> currentBand = origBand;
      
      // Apply tiling for each cache level, from outermost to innermost
      for (size_t lvl = 0; lvl < numLevels; ++lvl) {
        // Solve n-D model for this cache level
        SmallVector<unsigned, 8> tileSizes =
            solveTileSizesND(gamma, cacheElems[lvl]);

        llvm::errs() << "Cache level " << lvl << " tile sizes: ";
        for (unsigned ts : tileSizes) {
          llvm::errs() << ts << " ";
        }
        llvm::errs() << "\n";
        
        // Ensure tile sizes match band dimensionality
        tileSizes.resize(n, 32);  // pad with defaults if needed
        
        // Apply tiling to current innermost band
        SmallVector<AffineForOp, 8> newBand;
        if (failed(tilePerfectlyNested(currentBand, tileSizes, &newBand))) {
          llvm::errs() << "Tiling for cache level " << lvl << " failed.\n";
          break;  // Tiling failed, stop for this band
        }
        
        if (newBand.empty()) break;

        // Update current band to innermost loops of this level's tiling
        getPerfectlyNestedLoops(currentBand, newBand.back());
        if (currentBand.size() != n) break;  // Band structure changed
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


/// Solve n-D reuse model: sum_{i≠j} (alpha*gamma_i)^2 * (alpha*gamma_j)^2 = C
/// Returns tile sizes t_i = alpha * gamma_i for given cache capacity C.
static SmallVector<unsigned, 8>
solveTileSizesND(ArrayRef<double> gamma, double numElementsInCache) {
  size_t n = gamma.size();
  double sum_pairwise = 0.0;
  
  // Compute sum_{i≠j} gamma_i^2 * gamma_j^2
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      if (i != j) {
        sum_pairwise += gamma[i] * gamma[i] * gamma[j] * gamma[j];
      }
    }
  }
  
  double alpha2 = (sum_pairwise > 0.0) ? numElementsInCache / sum_pairwise : 1.0;
  double alpha = std::sqrt(std::max(alpha2, 1.0));
  
  SmallVector<unsigned, 8> tiles(n);
  for (size_t i = 0; i < n; ++i) {
    tiles[i] = std::max<unsigned>(1, (unsigned)std::floor(alpha * gamma[i]));
  }
  return tiles;
}


} // namespace tutorial
} // namespace mlir
