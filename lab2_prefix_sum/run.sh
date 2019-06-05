mkdir -p build
cd build
cmake -quiet ..
make
cd ..
cp build/lab2_prefix_sum .
./lab2_prefix_sum
