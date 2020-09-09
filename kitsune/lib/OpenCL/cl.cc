#include <CL/cl.hpp>
#include<iostream>
#include<stdint.h>
#include<stdbool.h>

extern "C" void __kitsune_opencl_init(); 
extern "C" void __kitsune_opencl_init_kernel(uint64_t id, uint64_t len, void* spirkernel); 
extern "C" void __kitsune_opencl_set_arg(int id, int argid, void* arg, uint32_t size, uint8_t mode); 
extern "C" void __kitsune_opencl_set_run_size(int id, uint64_t n); 
extern "C" void __kitsune_opencl_run_kernel(int id); 
extern "C" void __kitsune_opencl_finish(); 
extern "C" void __kitsune_opencl_mmap_marker(void* arg, uint64_t size); 
extern "C" void* __kitsune_opencl_mem_write(uint64_t id, void* arg, uint64_t size, uint8_t mode); 
extern "C" void __kitsune_opencl_mem_read(uint64_t id, void* ptr, void* buf, uint64_t size); 

using namespace std; 

void check(bool pred, std::string msg, int err=0){
  if(!pred){
    std::cerr << msg << std::endl;
    exit(-1);
  }
}

void checkCL(int err, std::string msg){
  if(err != CL_SUCCESS){
    std::cerr << msg << " errno: " << err << std::endl;
    exit(-1);
  }
}

/* Globals */
bool initialized = false; 
std::vector<cl::Platform> platforms;
std::vector<cl::Device> devices; 
cl::Kernel kernels[1024]; 
cl::Context context; 
cl::CommandQueue commandQueue; 
uint64_t sizes[1024]; 

void __kitsune_opencl_init(){
  if(!initialized){
    cl::Platform::get(&platforms); 
    check(platforms.size() > 0, "No opencl platforms found\n");
    platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
    check(devices.size() > 0, "No opencl gpu devices found\n"); 
    cl::Context c{devices[0]};
    context = c; 
    commandQueue = cl::CommandQueue{context, devices[0]};
    initialized = true; 
  }
}

void __kitsune_opencl_mmap_marker(void* arg, uint64_t size){}; 

void* __kitsune_opencl_mem_write(uint64_t id, void* ptr, uint64_t size, uint8_t mode){
  cl_int err; 
  cl_mem_flags mf = CL_MEM_COPY_HOST_PTR; 
  mf |= ((mode & 1) && (mode & 2)) ? CL_MEM_READ_WRITE :
        ((mode & 1) ? CL_MEM_READ_ONLY : 
        ((mode & 2) ? CL_MEM_WRITE_ONLY :
        CL_MEM_READ_WRITE)); 

  cl_mem mem = clCreateBuffer(context(), mf, size, ptr, &err);  
  checkCL(err, "clCreateBuffer failed");
  return (void*) mem; 
}

void __kitsune_opencl_mem_read(uint64_t id, void* ptr, void* buf, uint64_t size){
  cl_int err = clEnqueueReadBuffer(commandQueue(), (cl_mem)buf, true, 0, size, ptr, 0, NULL, NULL); 
  checkCL(err, "clEnqueueReadBuffer failed");
}

void __kitsune_opencl_init_kernel(uint64_t id, uint64_t len, void* spirkernel){
  cl::Program program {clCreateProgramWithIL(context(), spirkernel, len, nullptr)};   
  cl_int err = program.build(nullptr); 
  checkCL(err, "program build failed");
  kernels[id] = cl::Kernel{program, "kitsune_spirv_kernel", &err};
  checkCL(err, "kernel creation failed"); 
}

void __kitsune_opencl_set_arg(int id, int argid, void* arg, uint32_t size, uint8_t mode){
  clSetKernelArg(kernels[id](), argid, size, arg); 
}

void __kitsune_opencl_set_run_size(int id, uint64_t n){
  sizes[id] = n; 
}

void __kitsune_opencl_run_kernel(int id){
  cl_int err = commandQueue.enqueueNDRangeKernel(kernels[id], cl::NullRange, cl::NDRange{sizes[id]}); 
  checkCL(err, "enqueue kernel failed"); 
}

void __kitsune_opencl_finish(){
  commandQueue.finish(); 
}
