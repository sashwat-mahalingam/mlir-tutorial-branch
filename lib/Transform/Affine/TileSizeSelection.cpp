#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include <numeric>

namespace mlir {
    namespace polyTiling {

        #define GEN_PASS_DEF_TILESIZESELECTION
        #include "lib/Transform/Affine/Passes.h.inc"

        using namespace mlir;
        using namespace mlir::affine;

        struct TileSizeSelection
            : impl::TileSizeSelectionBase<TileSizeSelection> {
        
            // AffineNDMultiLevelTilingPass() {}

            void runOnOperation() {
                
                // Hard coding number of multi-cache levels to 3 for now 
                int numLevels = 3;
                // Hard code cache sizes for L1, L2, L3 cache
                std::vector<uint64_t> cacheBytes = {
                            8ULL*1024*1024,    // L3: 8MB  
                            1ULL*1024*1024,    // L2: 1MB
                            128ULL*1024};      // L1: 128KB
                uint64_t elemBytes = 4; 

                std::vector<double> cacheElems(numLevels);
                for (size_t lvl = 1; lvl <= numLevels; ++lvl) {
                cacheElems[lvl] = (double)cacheBytes[lvl] / (double)elemBytes;
                }

                // func::FuncOp func = getOperation();
                Operation *op = getOperation();

                op->walk([&](AffineForOp op) {
                    // tile size selection
                    SmallVector<AffineForOp, 8> origBand;
                    getPerfectlyNestedLoops(origBand, op);
                    
                    if (origBand.size() < 2) return;  // Need at least 2D for tiling
                    
                    size_t n = origBand.size();
                    
                    // Compute dimensional reuse γ once for this n-D band
                    SmallVector<double, 8> gamma = {0.5, 0.5, 1.0}; // hardcoded for now 
                    // = computeDimensionalReuse(origBand);
                    
                    // Current innermost band starts as original band
                    SmallVector<AffineForOp, 8> currentBand = origBand;

                    // print the current band loops (print constant upper bounds when available)
                    llvm::errs() << "Original band loops: ";
                    for (AffineForOp loop : currentBand) {
                        if (loop.hasConstantUpperBound()) {
                            // getConstantUpperBound() returns an int64_t
                            llvm::errs() << static_cast<double>(loop.getConstantUpperBound()) << " ";
                        } else {
                            llvm::errs() << "? ";
                        }
                    }
                    llvm::errs() << "\n";
                    
                    // Apply tiling for each cache level, from outermost to innermost
                    for (size_t lvl = 1; lvl <= numLevels; ++lvl) {
                        // Solve n-D model for this cache level
                        SmallVector<unsigned, 8> tileSizes =
                            solveTileSizesND(gamma, cacheElems[lvl]);

                        llvm::errs() << "Cache level " << lvl << " tile sizes: ";
                        for (unsigned ts : tileSizes) {
                        llvm::errs() << ts << " ";
                        }
                        llvm::errs() << "\n";
                        
                        // Ensure tile sizes match band dimensionality
                        tileSizes.resize(n, 0);  // pad with defaults if needed
                        
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
                    return;
                });
            }
            
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
            };
        };
    };
};
