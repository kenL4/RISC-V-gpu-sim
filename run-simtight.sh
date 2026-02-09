# This file can only be ran if you have already activated
# the sim in SIMTight. Please follow the instructions on
# the SIMTight GitHub page.

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

for APP in ${APPS[@]}; do
    echo "Running kernel: $APP"
    START_NS=$(date +%s%N)
    cd ./$APP
    ./RunSim
    cd ../../
    END_NS=$(date +%s%N)
    ELAPSED_MS=$(( (END_NS - START_NS) / 1000000 ))
    echo "WallTime_ms: $ELAPSED_MS"
done