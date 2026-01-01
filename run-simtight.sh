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
    cd ./$APP
    ./RunSim
    cd ../../
done