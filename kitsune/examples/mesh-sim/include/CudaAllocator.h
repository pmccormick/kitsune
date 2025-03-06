/**
 * @file CudaAllocator.h
 * @brief Custom allocator that allocates memory on the CUDA device.
 *
 * ================================================================================
 * Design Considerations:
 * ================================================================================
 * This allocator is designed to allocate memory using CUDA’s cudaMalloc/cudaFree
 * routines. It is intended for cases where data is used on the GPU exclusively.
 *
 * Pros:
 *   - Allocates memory directly in device memory, avoiding extra host-to-device copies.
 *   - Facilitates use with device containers (or custom containers) in CUDA code.
 *
 * Cons:
 *   - Memory allocated with cudaMalloc is not directly accessible by the host.
 *   - Requires CUDA headers and a CUDA-capable device.
 *   - Not suitable for code that runs entirely on the host.
 *
 * Future Considerations:
 *   - Integration with unified memory models or CUDA streams.
 *   - Error–handling improvements and logging.
 *
 * ================================================================================
 */

#ifndef CUDA_ALLOCATOR_H
#define CUDA_ALLOCATOR_H

#include <cuda_runtime.h>
#include <cstddef>
#include <stdexcept>
#include <limits>
#include <new>
#include <memory>

template <typename T>
class CudaAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    
    template <typename U>
    struct rebind {
        using other = CudaAllocator<U>;
    };
    
    CudaAllocator() noexcept {}
    template <typename U>
    CudaAllocator(const CudaAllocator<U>&) noexcept {}
    
    pointer allocate(size_type n, const void* hint = 0) {
        (void)hint;
        if (n > std::numeric_limits<size_type>::max() / sizeof(T))
            throw std::bad_alloc();
        
        T* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, n * sizeof(T));
        if (err != cudaSuccess)
            throw std::bad_alloc();
        return ptr;
    }
    
    void deallocate(pointer p, size_type n) noexcept {
        (void)n;
        cudaFree(p);
    }
    
    template <typename U>
    bool operator==(const CudaAllocator<U>&) const noexcept { return true; }
    template <typename U>
    bool operator!=(const CudaAllocator<U>&) const noexcept { return false; }
};

#endif // CUDA_ALLOCATOR_H



