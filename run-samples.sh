# This list is literally taken directly from SIMTight LOL
APPS=(
  Samples/VecAdd
  Samples/Histogram
  Samples/Reduce
  Samples/Scan
  Samples/Transpose
  Samples/MatVecMul
  Samples/MatMul
  Samples/BitonicSortSmall
  Samples/BitonicSortLarge
  Samples/SparseMatVecMul
  InHouse/BlockedStencil
  InHouse/StripedStencil
  InHouse/VecGCD
  InHouse/MotionEst
)

make

for APP in ${APPS[@]}; do
    echo "Running kernel: $APP"
    ./build/RISCVGpuSim $APP/app.elf -s
done