#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>

#include "./cl.hpp"

int main() {
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    std::vector<cl::Kernel> kernels;

    try {
        // create platform
        cl::Platform::get(&platforms);
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

        // create context
        cl::Context context(devices);

        // create command queue
        cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

        // load opencl source
        std::ifstream cl_file("convolution.cl");
        std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, std::make_pair(cl_string.c_str(),
                                                      cl_string.length() + 1));

        // create program
        cl::Program program(context, source);

        try {
            // compile opencl source
            program.build(devices);
        } catch (cl::Error const &e) {
            std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
            std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
            exit(1);
        }

        std::ifstream in("input.txt");
        std::ofstream out("output.txt");

        float next;
        int n, m;
        in >> n >> m;

        std::vector<float> a(n * n);
        std::vector<float> b(m * m);
        std::vector<float> c(n * n);

        for (int row = 0; row < n; row++) {
            for (int column = 0; column < n; column++) {
                in >> a[row * n + column];
            }
        }

        for (int row = 0; row < m; row++) {
            for (int column = 0; column < m; column++) {
                in >> b[row * m + column];
            }
        }

        size_t const block_size = 16;
        size_t const rounded_n = ((n + block_size - 1) / block_size) * block_size;

        // allocate device buffer to hold message
        cl::Buffer dev_a(context, CL_MEM_READ_ONLY, sizeof(float) * a.size());
        cl::Buffer dev_b(context, CL_MEM_READ_ONLY, sizeof(float) * b.size());
        cl::Buffer dev_c(context, CL_MEM_READ_WRITE, sizeof(float) * c.size());

        // copy from cpu to gpu
        queue.enqueueWriteBuffer(dev_a, CL_TRUE, 0, sizeof(float) * a.size(), &a[0]);
        queue.enqueueWriteBuffer(dev_b, CL_TRUE, 0, sizeof(float) * b.size(), &b[0]);

        // load named kernel from opencl source
        cl::Kernel kernel_convolution(program, "get_convolution");
        kernel_convolution.setArg(0, dev_a);
        kernel_convolution.setArg(1, dev_b);
        kernel_convolution.setArg(2, dev_c);
        kernel_convolution.setArg(3, cl_int(n));
        kernel_convolution.setArg(4, cl_int(m));

        queue.enqueueNDRangeKernel(kernel_convolution, cl::NullRange, cl::NDRange(rounded_n, rounded_n),
                cl::NDRange(block_size, block_size));
        queue.finish();
        queue.enqueueReadBuffer(dev_c, CL_TRUE, 0, sizeof(float) * c.size(), &c[0]);

        out << std::fixed;
        for (int row = 0; row < n; row++) {
            for (int column = 0; column < n; column++) {
                out << std::setprecision(3) << c[row * n + column] << (column == n - 1 ? "" : " ");
            }
            out << std::endl;
        }

        in.close();
        out.close();
    }
    catch (cl::Error const &e) {
        std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
        exit(1);
    }

    return 0;
}