#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassOptions.h"
#include <numeric>
#include <cstdlib>
#include <string>

namespace mlir {
    namespace polyTiling {

        #define GEN_PASS_DEF_TILESIZESELECTION
        #include "lib/Transform/Affine/Passes.h.inc"

        using namespace mlir;
        using namespace mlir::affine;

        struct TileSizeSelection 
            : impl::TileSizeSelectionBase<TileSizeSelection> {

            void runOnOperation() {
                // get num levels from environment
                const char* numLevelsEnv = std::getenv("NUM_TILE_LEVELS");
                const char* numBandsEnv = std::getenv("NUM_BANDS");
                if (!numLevelsEnv || !numBandsEnv) {
                    llvm::errs() << "Error: NUM_TILE_LEVELS or NUM_BANDS environment variables not set\n";
                    return;
                }
                std::string numLevelsStr = numLevelsEnv;
                std::string numBandsStr = numBandsEnv;
                int numLevels = std::stoi(numLevelsStr);
                int numBands = std::stoi(numBandsStr);
                
                std::vector<std::vector<int64_t>> tileLevels;
                for (unsigned i = 0; i < numLevels; ++i) {
                    std::vector<int64_t> tileLevel;
                    for (unsigned j = 0; j < numBands; ++j) {
                        std::string envVarName = "TILE_LEVEL_" + std::to_string(i) + "_" + std::to_string(j);
                        const char* levelijEnv = std::getenv(envVarName.c_str());
                        if (!levelijEnv) {
                            llvm::errs() << "Error: Environment variable " << envVarName << " not set\n";
                            return;
                        }
                        std::string levelijStr = levelijEnv;
                        int64_t tileijSize = std::stoi(levelijStr);
                        tileLevel.push_back(tileijSize);
                    }
                    tileLevels.push_back(tileLevel);
                }

                Operation *op = getOperation();

                op->walk([&](AffineForOp rootLoop) {
                    // If this loop is nested inside another AffineForOp, skip it here
                    // and let the outermost walk starting point perform tiling. This
                    // ensures we only start recursive tiling once per outer band.
                    if (rootLoop.getOperation()->getParentOfType<AffineForOp>())
                        return;

                    // Compute the perfectly nested loop band starting at this walk entry.
                    SmallVector<AffineForOp, 8> origBand;
                    getPerfectlyNestedLoops(origBand, rootLoop);

                    if (origBand.size() < 2) return;  // Need at least 2D for tiling

                    size_t n = origBand.size();
                    // print n

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
                        SmallVector<unsigned, 8> tileSizes;

                        if (numBands < bandsToTile) {
                            llvm::errs() << "Not enough band values given to tile, return " << numBands << " < " << bandsToTile << "\n";
                            return false;
                        }

                        for (unsigned j = numBands - bandsToTile; j < numBands; ++j) {
                            tileSizes.push_back((unsigned) tileLevels[lvl][j]);
                        }

                        for (unsigned ts : tileSizes) llvm::errs() << ts << " ";
                        llvm::errs() << "\n";

                        SmallVector<AffineForOp, 8> truncBands;
                        for (size_t i = 0; i < bandsToTile; ++i)
                            truncBands.push_back(currentBand[i]);

                        SmallVector<AffineForOp, 8> newBand;
                        if (failed(tilePerfectlyNested(truncBands, tileSizes, &newBand))) {
                            llvm::errs() << "Tiling for cache level " << lvl << " failed.\n";
                            return false;
                        }

                        if (newBand.empty()) return true;
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
        };
    };
};
