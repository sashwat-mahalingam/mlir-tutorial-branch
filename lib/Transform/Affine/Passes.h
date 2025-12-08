#ifndef LIB_TRANSFORM_AFFINE_PASSES_H_
#define LIB_TRANSFORM_AFFINE_PASSES_H_

#include "lib/Transform/Affine/PreTileAnalysis.h"
#include "lib/Transform/Affine/RaiseToAffine.h" // <-- add this

namespace mlir {
namespace polyTiling {

#define GEN_PASS_REGISTRATION
#include "lib/Transform/Affine/Passes.h.inc"

}  // namespace polyTiling
}  // namespace mlir

#endif  // LIB_TRANSFORM_AFFINE_PASSES_H_
