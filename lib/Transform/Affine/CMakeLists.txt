add_mlir_library(Project583Passes
    RaiseToAffine.cpp
    AffineFullUnroll.cpp
    AffineFullUnrollPatternRewrite.cpp 

    ${PROJECT_SOURCE_DIR}/lib/Transform/Affine/
    ADDITIONAL_HEADER_DIRS

    DEPENDS
    MLIRFullProject583Passes

    LINK_LIBS PUBLIC
)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Affine)
add_public_tablegen_target(MLIRFullProject583Passes)
add_mlir_doc(Passes AffinePasses ./ -gen-pass-doc)
