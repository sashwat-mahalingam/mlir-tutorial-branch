#ifndef LIB_TRANSFORM_AFFINE_TILESIZESELECTIONPASS_H_
#define LIB_TRANSFORM_AFFINE_TILESIZESELECTIONPASS_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace affine {

#define GEN_PASS_DECL_TILESIZESELECTIONPASS
#include "lib/Transform/Affine/Passes.h.inc"

}  // namespace affine
}  // namespace mlir

#endif  // LIB_TRANSFORM_AFFINE_TILESIZESELECTIONPASS_H_
