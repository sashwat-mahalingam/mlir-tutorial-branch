#ifndef LIB_TRANSFORM_AFFINE_RAISE_TO_AFFINE_PASS_H_
#define LIB_TRANSFORM_AFFINE_RAISE_TO_AFFINE_PASS_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace affine {

// Factory function for the pass
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> createRaiseMemrefToAffine();

} // namespace affine
} // namespace mlir

#endif // LIB_TRANSFORM_AFFINE_RAISE_TO_AFFINE_PASS_H_
