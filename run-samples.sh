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
    START_NS=$(date +%s%N)
    ./build/RISCVGpuSim $APP/app.elf -s
    END_NS=$(date +%s%N)
    ELAPSED_MS=$(( (END_NS - START_NS) / 1000000 ))
    echo "WallTime_ms: $ELAPSED_MS"
done