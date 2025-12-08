get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set (LIBS
    ${dialect_libs}
    ${conversion_libs}
    Project583Passes
    MLIROptLib
    MLIRPass
)

add_llvm_executable(mlir-poly-tiling-opt mlir-poly-tiling-opt.cpp)

llvm_update_compile_flags(mlir-poly-tiling-opt)
target_link_libraries(mlir-poly-tiling-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(mlir-poly-tiling-opt)
