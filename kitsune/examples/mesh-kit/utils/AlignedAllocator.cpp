#include "AlignedAllocator.h"
#include <vector>
#include <iostream>

// This file demonstrates an instantiation of AlignedAllocator for double.
// Because the allocator is templated and header-only, this .cpp can be used
// to test that the allocator compiles and behaves as expected.
int main() {
    std::vector<double, AlignedAllocator<double, 64>> vec;
    vec.resize(100);
    
    // Verify that the returned pointer is aligned.
    double* ptr = vec.data();
    if (reinterpret_cast<std::uintptr_t>(ptr) % 64 == 0)
        std::cout << "AlignedAllocator: Memory is 64-byte aligned." << std::endl;
    else
        std::cout << "AlignedAllocator: Memory is NOT properly aligned!" << std::endl;
    
    return 0;
}

