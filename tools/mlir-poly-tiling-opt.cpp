// Minimal mlir-opt with affine pass
#include "lib/Transform/Affine/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();
  
  mlir::polyTiling::registerAffinePasses();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "MLIR Polynomial Tiling Pass Driver", registry));
}