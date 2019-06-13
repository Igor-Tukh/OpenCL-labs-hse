mkdir -p build
cd build
cmake -quiet ..
make
cd ..
cp build/lab1_convolution .
./lab1_convolution