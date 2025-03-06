/**
 * @file PoolAllocator.h
 * @brief A slab-based pool allocator for aligned CPU memory, working entirely in bytes.
 *
 * ================================================================================
 * Design Considerations:
 * ================================================================================
 * This pool allocator is designed for scientific simulations where the total memory
 * requirements for Field data (or similar) are known at startup. It pre-allocates large
 * contiguous blocks of memory (slabs) with a fixed alignment (e.g., 64 bytes) and then
 * subdivides each slab into fixed–sized blocks.
 *
 * Key features include:
 *   - **Byte-Oriented Interface:**  
 *     Unlike templated allocators, this allocator works purely in terms of bytes. All
 *     allocation requests are in byte units, so you only need one allocator instance for
 *     all your data types.
 *
 *   - **Aligned Memory:**  
 *     The allocator guarantees that each allocated block is aligned to a specified boundary
 *     (default 64 bytes) to support SIMD vectorization and cache–friendly memory access.
 *
 *   - **Slab-Based Pool with Recycling:**  
 *     Memory is allocated in slabs of a fixed size (default 4096 bytes). Each slab is divided
 *     into blocks of a fixed block size. Freed blocks are returned to a free list, and when
 *     an entire slab becomes unused, it can be recycled (released or re–used) to minimize
 *     fragmentation.
 *
 *   - **Function-Call Interface Option:**  
 *     A simple wrapper interface (pool_alloc() / pool_free()) is provided so that the pool
 *     allocator can be used without explicitly managing a C++ object.
 *
 * ================================================================================
 * Pros and Cons:
 * ================================================================================
 * Pros:
 *   - Significantly reduces allocation overhead when many small objects of fixed size are
 *     needed.
 *   - Guarantees memory alignment, which is critical for high-performance numerical kernels.
 *   - Recycling freed blocks minimizes fragmentation and maximizes reuse of allocated memory.
 *
 * Cons:
 *   - Designed for fixed–size allocations. If allocation sizes vary widely, some memory may
 *     be wasted.
 *   - Thread safety is provided via a mutex; in extremely concurrent environments, this may
 *     become a bottleneck. Advanced designs might use lock-free or thread-local pools.
 *   - Increased code complexity compared to using std::vector directly.
 *
 * Future Enhancements:
 *   - Extend to support variable–sized allocations (perhaps using a buddy allocator scheme).
 *   - Integrate custom allocators for GPU-managed memory (e.g., using cudaMallocManaged).
 *   - Add optional prefetching and NUMA–aware placement for multi–core systems.
 *
 * Literature for further study:
 *   - "The Slab Allocator: An Object-Caching Kernel Memory Allocator" by Jeff Bonwick.
 *   - Linux kernel documentation on slab and SLUB allocators.
 *
 * ================================================================================
 */

#ifndef POOL_ALLOCATOR_H
#define POOL_ALLOCATOR_H

#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#include <mutex>
#include <limits>
#include <cassert>
#include <cstdint>

// Cross-platform aligned allocation:
#if defined(_MSC_VER)
  #include <malloc.h>
#endif

class PoolAllocator {
public:
    /**
     * @brief Construct a pool allocator.
     *
     * @param blockSize The fixed size (in bytes) for each allocation.
     * @param slabSize The size (in bytes) of each slab. Default is 4096 bytes.
     * @param alignment The desired memory alignment. Default is 64 bytes.
     *
     * Note: blockSize must be at least sizeof(void*) to store free list pointers.
     */
    PoolAllocator(std::size_t blockSize, std::size_t slabSize = 4096, std::size_t alignment = 64);
    
    ~PoolAllocator();

    /**
     * @brief Allocate a block of memory.
     *
     * @param n The size in bytes to allocate. Must be <= blockSize.
     * @return Pointer to the allocated memory.
     *
     * If n is not equal to blockSize, this allocator will allocate a full block
     * (of size blockSize) and return a pointer to it.
     */
    void* allocate(std::size_t n);
    
    /**
     * @brief Deallocate a block of memory.
     *
     * @param p Pointer to the memory to deallocate.
     * @param n The size in bytes of the block (must be <= blockSize).
     *
     * The block is returned to the free list for future reuse.
     */
    void deallocate(void* p, std::size_t n);
    
    /**
     * @brief Get the total capacity (number of blocks) currently managed.
     */
    std::size_t totalCapacity() const;

private:
    // Internal structure for a slab.
    struct Slab {
        void* memory;           // Pointer to the beginning of the slab.
        std::size_t capacity;   // Number of blocks in this slab.
        std::size_t used;       // Number of blocks currently allocated.
        
        Slab(void* mem, std::size_t cap) : memory(mem), capacity(cap), used(0) {}
    };

    // Free list node. Stored in freed block.
    struct FreeNode {
        FreeNode* next;
    };

    // Data members.
    const std::size_t m_blockSize;  // Size of each block (in bytes).
    const std::size_t m_slabSize;   // Size of each slab (in bytes).
    const std::size_t m_alignment;  // Memory alignment requirement.
    
    std::vector<Slab> m_slabs;      // Pool of slabs.
    FreeNode* m_freeList;           // Global free list for blocks.
    mutable std::mutex m_mtx;       // Mutex for thread safety.

    // Allocate a new slab and add its blocks to the free list.
    void allocateNewSlab();
    
    // Determine the number of blocks that can fit in a slab.
    std::size_t blocksPerSlab() const;
    
    // Check if a pointer belongs to a given slab.
    bool pointerInSlab(void* p, const Slab& slab) const;
    
    // Optionally, reclaim slabs that are entirely free.
    void tryReclaimSlabs();
};

#endif // POOL_ALLOCATOR_H


