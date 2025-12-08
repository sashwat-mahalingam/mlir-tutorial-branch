export PATH=$HOME/Polygeist/build/bin:$PATH

cgeist benchmarks/matmul.c -I $HOME/llvm-mlir-pgeist/lib/clang/18/include \
    --function=matmul -S | polygeist-opt --raise-scf-to-affine --polygeist-mem2reg -o benchmarks/matmul_affine_unraised.mlir

./build-ninja/tools/mlir-poly-tiling-opt benchmarks/matmul_affine_unraised.mlir --raise-to-affine -o benchmarks/matmul_affine.mlir
./build-ninja/tools/mlir-poly-tiling-opt benchmarks/matmul_affine.mlir --pre-tile-analysis -o benchmarks/matmul_affine.mlir

# mlir-opt matmul_affine.mlir \
#     --affine-loop-tile="tile-sizes=32,64,38" \
#     --canonicalize \
#     --cse \
#     -o matmul_tiled.mlir
    
# polygeist-opt matmul_tiled.mlir --lower-affine | mlir-opt --convert-scf-to-cf | mlir-opt --convert-to-llvm | mlir-opt \
#     --reconcile-unrealized-casts | mlir-translate --mlir-to-llvmir -o matmul.ll