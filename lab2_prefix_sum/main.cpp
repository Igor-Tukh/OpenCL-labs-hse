#define __CL_ENABLE_EXCEPTIONS

#include <OpenCL/cl.h>
#include "cl.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <iterator>
#include <iomanip>
#include <assert.h>

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
        std::ifstream cl_file("scan.cl");
        std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, std::make_pair(cl_string.c_str(),
                                                      cl_string.length() + 1));

        // create program
        cl::Program program(context, source);

        // compile opencl source
        program.build(devices);

        int N = 0;
        std::ifstream in("input.txt");
        in >> N;

        // create a message to send to kernel
        std::vector<float> input(N);
        std::vector<float> output(N, 0.0);
        for (size_t i = 0; i < N; ++i) {
            in >> input[i];
        }
        in.close();

        size_t const block_size = 32;
        size_t const blocks_amount = (N + block_size - 1) / block_size;
        size_t const N_size = blocks_amount * block_size;

        // allocate device buffer to hold message
        cl::Buffer dev_input(context, CL_MEM_READ_ONLY, sizeof(float) * N);
        cl::Buffer dev_inter(context, CL_MEM_READ_WRITE, sizeof(float) * blocks_amount);
        cl::Buffer dev_sums(context, CL_MEM_READ_WRITE, sizeof(float) * blocks_amount);
        cl::Buffer dev_tmp(context, CL_MEM_READ_WRITE, sizeof(float));
        cl::Buffer dev_output(context, CL_MEM_READ_WRITE, sizeof(float) * N);

        // copy from cpu to gpu
        queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, sizeof(float) * N, &input[0]);
        queue.finish();

        cl::Kernel kernel_b(program, "scan_blelloch");
        cl::KernelFunctor scan_b(kernel_b, queue, cl::NullRange, cl::NDRange(N_size), cl::NDRange(block_size));
        cl::Event event = scan_b(dev_input, dev_output, cl::__local(sizeof(float) * block_size), dev_inter, N);

        cl::Kernel kernel_sum(program, "sum_with_inter");
        cl::KernelFunctor sum(kernel_sum, queue, cl::NullRange, cl::NDRange(N_size), cl::NDRange(block_size));

        event.wait();
        cl_ulong start_time = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();


        event = scan_b(dev_inter, dev_sums, cl::__local(sizeof(float) * block_size), dev_tmp, blocks_amount);
        event.wait();
        event = sum(dev_output, dev_sums, N);
        event.wait();

        cl_ulong end_time = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
        cl_ulong elapsed_time = end_time - start_time;

        queue.enqueueReadBuffer(dev_output, CL_TRUE, 0, sizeof(float) * N, &output[0]);
        std::ofstream out("output.txt");
        for (const auto& val: output) {
            out << std::fixed << std::setprecision(3) << val << " ";
        }
        out.close();

        std::cout << std::setprecision(2) << "Total time: " << elapsed_time / 1000000.0 << " ms" << std::endl;
    }
    catch (cl::Error e) {
        std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
    }

    return 0;
}