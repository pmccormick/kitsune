#include <CL/cl.hpp>
#include<iostream>

using namespace std; 

void check(bool pred, std::string msg){
  if(!pred){
    std::cerr << msg << std::endl;
    exit(-1);
  }
}

void run_kernel(void* spirkernel, size_t len, size_t n){
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms); 
  check(platforms.size() > 0, "No opencl platforms found\n");
  std::vector<cl::Device> devices; 
  platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
  check(devices.size() > 0, "No opencl gpu devices found\n"); 
  cl::Context context{devices[0]};
  auto commandQueue  = cl::CommandQueue{context, devices[0]};
  cl::Program program{clCreateProgramWithIL(context(), spirkernel, 1, nullptr)};   
  program.build(nullptr); 
  auto kernel = cl::Kernel{program, "kitsune_spirv_kernel"}; 
  commandQueue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange{n}); 
  commandQueue.finish(); 
}
