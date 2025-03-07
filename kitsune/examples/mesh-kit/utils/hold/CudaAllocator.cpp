#include "CudaAllocator.h"
#include <vector>
#include <iostream>
#include <cuda_runtime.h>

int main() {
    try {
        // Instantiate a vector of doubles using the CudaAllocator.
        // Note: This vector is intended for device memory and cannot be directly accessed on host.
        std::vector<double, CudaAllocator<double>> d_vec;
        d_vec.resize(100);
        std::cout << "CudaAllocator: Successfully allocated 100 doubles on the device." << std::endl;
        
        // Freeing is handled automatically by the vector's destructor.
    } catch (std::bad_alloc&) {
        std::cerr << "CudaAllocator: Allocation failed!" << std::endl;
        return 1;
    }
    return 0;
}

