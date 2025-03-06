#ifndef POOL_ALLOC_INTERFACE_H
#define POOL_ALLOC_INTERFACE_H

#include "PoolAllocator.h"
#include <cstddef>

// Global pool allocator instance for byte allocations.
// For example, assume we want fixed blocks of 256 bytes, and slabs of 64KB.
static PoolAllocator globalPoolAllocator(256, 65536, 64);

/**
 * @brief Allocate memory from the global pool.
 * @param n Number of bytes to allocate (should be <= 256).
 * @return Pointer to allocated memory.
 */
inline void* pool_alloc(std::size_t n) {
    // For simplicity, if n != 256, we round up to 256.
    return globalPoolAllocator.allocate(n);
}

/**
 * @brief Deallocate memory back to the global pool.
 * @param p Pointer to memory.
 * @param n Number of bytes originally allocated.
 */
inline void pool_free(void* p, std::size_t n) {
    globalPoolAllocator.deallocate(p, n);
}

#endif // POOL_ALLOC_INTERFACE_H


