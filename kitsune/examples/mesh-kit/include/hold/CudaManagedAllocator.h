/**
 * @file CudaManagedAllocator.h
 * @brief Custom allocator that allocates unified (managed) memory using CUDA.
 *
 * ================================================================================
 * Design Considerations:
 * ================================================================================
 * This allocator uses cudaMallocManaged to allocate memory that is accessible both
 * on the host and the device. It enables a unified memory model, simplifying data
 * movement between host and GPU.
 *
 * Pros:
 *   - Memory is automatically migrated between host and device.
 *   - Simplifies development by allowing a single pointer to be used on both sides.
 *
 * Cons:
 *   - Performance may be lower than using separate host and device allocations,
 *     especially for data-intensive operations with frequent transfers.
 *   - Requires CUDA 6.0 or later and may not be optimal for all use cases.
 *
 * Future Considerations:
 *   - Tuning data placement and prefetching can help improve performance.
 *   - Integration with CUDA streams for asynchronous data management.
 *
 * ================================================================================
 */

#ifndef CUDA_MANAGED_ALLOCATOR_H
#define CUDA_MANAGED_ALLOCATOR_H

#include <cuda_runtime.h>
#include <cstddef>
#include <stdexcept>
#include <limits>
#include <new>
#include <memory>

template <typename T>
class CudaManagedAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    
    template <typename U>
    struct rebind {
        using other = CudaManagedAllocator<U>;
    };
    
    CudaManagedAllocator() noexcept {}
    template <typename U>
    CudaManagedAllocator(const CudaManagedAllocator<U>&) noexcept {}
    
    pointer allocate(size_type n, const void* hint = 0) {
        (void)hint;
        if (n > std::numeric_limits<size_type>::max() / sizeof(T))
            throw std::bad_alloc();
        
        T* ptr = nullptr;
        cudaError_t err = cudaMallocManaged(&ptr, n * sizeof(T));
        if (err != cudaSuccess)
            throw std::bad_alloc();
        return ptr;
    }
    
    void deallocate(pointer p, size_type n) noexcept {
        (void)n;
        cudaFree(p);
    }
    
    template <typename U>
    bool operator==(const CudaManagedAllocator<U>&) const noexcept { return true; }
    template <typename U>
    bool operator!=(const CudaManagedAllocator<U>&) const noexcept { return false; }
};

#endif // CUDA_MANAGED_ALLOCATOR_H

