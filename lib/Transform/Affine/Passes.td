#ifndef LIB_TRANSFORM_AFFINE_PASSES_TD_
#define LIB_TRANSFORM_AFFINE_PASSES_TD_

include "mlir/Pass/PassBase.td"

def PreTileAnalysis : Pass<"pre-tile-analysis"> {
  let summary = "Analyze permutability, reuse factors, memory footprint, and heuristics needed for polynomial tile size selection";
  let description = [{
    Analyze permutability, reuse factors, memory footprint, and heuristics needed for polynomial tile size selection.
  }];
  let dependentDialects = ["mlir::affine::AffineDialect"];
}

def RaiseToAffine : Pass<"raise-to-affine"> {
  let summary = "raise memref to affine memory access when possible";
  let description = [{
    Raise MemRef to affine memory access when possible.
  }];
  let dependentDialects = ["mlir::affine::AffineDialect", "mlir::func::FuncDialect", "mlir::memref::MemRefDialect"];
}

#endif  // LIB_TRANSFORM_AFFINE_PASSES_TD_
