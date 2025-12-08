#ifndef LIB_TRANSFORM_AFFINE_TILESIZESELECTION_H_
#define LIB_TRANSFORM_AFFINE_TILESIZESELECTION_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace polyTiling {

#define GEN_PASS_DECL_TILESIZESELECTION
#include "lib/Transform/Affine/Passes.h.inc"

}  // namespace polyTiling
}  // namespace mlir

#endif  // LIB_TRANSFORM_AFFINE_TILESIZESELECTION_H_
