#include <CL/cl.h>
#include<math.h>
#include<stdio.h>
#include<stdint.h>
#include<stdbool.h>
#include<sys/types.h>
#include<sys/stat.h>
#include<unistd.h>

void __kitsune_opencl_init(); 
void __kitsune_opencl_init_kernel(uint64_t id, uint64_t len, void* spirkernel); 
void __kitsune_opencl_set_arg(int id, int argid, void* arg, uint32_t size, uint8_t mode); 
void __kitsune_opencl_set_run_size(int id, uint64_t n); 
void __kitsune_opencl_run_kernel(int id); 
void __kitsune_opencl_finish(); 
void __kitsune_opencl_mmap_marker(void* arg, uint64_t size); 
void* __kitsune_opencl_mem_write(uint64_t id, void* arg, uint64_t size, uint8_t mode); 
void __kitsune_opencl_mem_read(uint64_t id, void* ptr, void* buf, uint64_t size); 

int main(){
  FILE *f; 
  f = fopen("kernel.spv", "rb"); 
  int fd = fileno(f); 
  struct stat inf;
  fstat(fd, &inf);
  uint8_t *buf = (uint8_t*)malloc(inf.st_size); 
  int nbytesread = fread(buf, 1, inf.st_size, f); 
  
  int n = 1024, nbytes = n*sizeof(double); 
  double *a = (double*)malloc(nbytes);
  double *b = (double*)malloc(nbytes);
  double *c = (double*)malloc(nbytes);
  for(int i=0; i<n; i++){
    a[i] = sin(i)*sin(i);
    b[i] = cos(i)*cos(i);
    c[i] = 5; 
  }

  __kitsune_opencl_init();
  __kitsune_opencl_init_kernel(0, nbytesread, buf); 
  void* ab = __kitsune_opencl_mem_write(0, (void*)a, nbytes, 1); 
  void* bb = __kitsune_opencl_mem_write(0, (void*)b, nbytes, 1); 
  void* cb = __kitsune_opencl_mem_write(0, (void*)c, nbytes, 2); 
  __kitsune_opencl_set_arg(0, 0, &ab, sizeof(cl_mem), 1); 
  __kitsune_opencl_set_arg(0, 1, &bb, sizeof(cl_mem), 1); 
  __kitsune_opencl_set_arg(0, 2, &cb, sizeof(cl_mem), 2); 
  __kitsune_opencl_set_run_size(0, n); 
  __kitsune_opencl_run_kernel(0); 

  __kitsune_opencl_mem_read(0, (void*)c, cb, nbytes); 
  __kitsune_opencl_finish(); 

  for(int i=0; i<n; i++){
    if(c[i] != a[i] + b[i]) {
      printf("failure!\n");
      exit(-1);
    }
  }
  printf("success\n"); 
}

