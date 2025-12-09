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
                // Fill cache element capacities for each level (0-based indices).
                for (size_t lvl = 0; lvl < (size_t)numLevels; ++lvl) {
                    cacheElems[lvl] = (double)cacheBytes[lvl] / (double)elemBytes;
                }

                Operation *op = getOperation();

                op->walk([&](AffineForOp rootLoop) {
                    // If this loop is nested inside another AffineForOp, skip it here
                    // and let the outermost walk starting point perform tiling. This
                    // ensures we only start recursive tiling once per outer band.
                    if (rootLoop.getOperation()->getParentOfType<AffineForOp>())
                        return;
                    llvm::errs() << "Start \n";

                    // Compute the perfectly nested loop band starting at this walk entry.
                    SmallVector<AffineForOp, 8> origBand;
                    getPerfectlyNestedLoops(origBand, rootLoop);

                    if (origBand.size() < 2) return;  // Need at least 2D for tiling

                    size_t n = origBand.size();
                    // print n
                    llvm::errs() << "Processing band of dimensionality n = " << n << "\n";

                    // Compute dimensional reuse γ once for this n-D band.
                    // Hardcode a neutral reuse pattern (0.5) of length n for now.
                    SmallVector<double, 8> gamma(n, 0.5);

                    // Recursive tiling: at each cache level, tile the current band and
                    // recurse only on the sub-band corresponding to dimensions that
                    // were actually tiled (tile size > 1).
                    std::function<bool(SmallVector<AffineForOp, 8> &, size_t, size_t)> tileRec;
                    tileRec = [&](SmallVector<AffineForOp, 8> &currentBand, size_t lvl, size_t bandsToTile) -> bool {
                        if (currentBand.size() < 2){
                            llvm::errs() << "Band too small, return " << currentBand.size() << "\n";
                            return true;
                        } 
                        if (lvl >= (size_t)numLevels) return true;

                        // llvm::errs() << "Enter level " << lvl << "\n";

                        // Solve n-D model for this cache level using only the active
                        // dimensions (length of gamma should match n for the band).
                        SmallVector<unsigned, 8> tileSizes = solveTileSizesND(gamma, cacheElems[lvl]);
                        llvm::errs() << "Cache level " << lvl << " tile sizes: ";
                        for (unsigned ts : tileSizes) llvm::errs() << ts << " ";
                        llvm::errs() << "\n";

                        SmallVector<AffineForOp, 8> truncBands;
                        for (size_t i = 0; i < bandsToTile; ++i)
                            truncBands.push_back(currentBand[i]);
                        // llvm::errs() << "Truncated band size: " << truncBands.size() << "\n";

                        // Ensure tile sizes match band dimensionality; pad with a small default
                        // (avoid zeros, which make tiling invalid).
                        // tileSizes.resize(currentBand.size(), 32);

                        // llvm::errs() << "Resized tile sizes: ";
                        // for (unsigned ts : tileSizes) llvm::errs() << ts << " ";
                        // llvm::errs() << "\n";

                        SmallVector<AffineForOp, 8> newBand;
                        if (failed(tilePerfectlyNested(truncBands, tileSizes, &newBand))) {
                            llvm::errs() << "Tiling for cache level " << lvl << " failed.\n";
                            return false;
                        }

                        if (newBand.empty()) return true;
                        // llvm::errs() << "Size of new band: " << newBand.size() << "\n";
                        llvm::errs() << "\n";

                        return tileRec(newBand, lvl + 1, bandsToTile);
                    };

                    // Start recursion from the original band.
                    SmallVector<AffineForOp, 8> startBand = origBand;
                    tileRec(startBand, 0, n);
                    llvm::errs() << "\n";
                    return;
                });
            }
            
            /// Solve n-D reuse model: sum_{i≠j} (alpha*gamma_i)^2 * (alpha*gamma_j)^2 = C
            /// Returns tile sizes t_i = alpha * gamma_i for given cache capacity C.
            static SmallVector<unsigned, 8>
            solveTileSizesND(ArrayRef<double> gamma, double numElementsInCache) {
                size_t n = gamma.size();
                double sum_pairwise = 0.0;
                
                // Compute sum_{i≠j} gamma_i * gamma_j
                for (size_t i = 0; i < n; ++i) {
                    for (size_t j = 0; j < n; ++j) {
                    if (i != j) {
                        sum_pairwise += gamma[i] * gamma[j];
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
