#include "PoolAllocator.h"
#include <new>
#include <algorithm>
#include <iostream>

// Constructor.
PoolAllocator::PoolAllocator(std::size_t blockSize, std::size_t slabSize, std::size_t alignment)
    : m_blockSize(std::max(blockSize, sizeof(FreeNode*))), m_slabSize(slabSize),
      m_alignment(alignment), m_freeList(nullptr) {
    // Optionally, pre-allocate one slab.
    allocateNewSlab();
}

// Destructor: free all slabs.
PoolAllocator::~PoolAllocator() {
    for (auto& slab : m_slabs) {
#if defined(_MSC_VER)
        _aligned_free(slab.memory);
#else
        free(slab.memory);
#endif
    }
}

// Determine how many blocks fit in a slab.
std::size_t PoolAllocator::blocksPerSlab() const {
    return m_slabSize / m_blockSize;
}

// Allocate a new slab.
void PoolAllocator::allocateNewSlab() {
    std::size_t blocks = blocksPerSlab();
    if (blocks == 0)
        blocks = 1;
    
    void* ptr = nullptr;
#if defined(_MSC_VER)
    ptr = _aligned_malloc(m_slabSize, m_alignment);
    if (!ptr)
        throw std::bad_alloc();
#else
    if (posix_memalign(&ptr, m_alignment, m_slabSize) != 0)
        throw std::bad_alloc();
#endif
    // Create a new slab.
    m_slabs.emplace_back(ptr, blocks);
    
    // Partition the slab into blocks and push them onto the free list.
    char* base = static_cast<char*>(ptr);
    for (std::size_t i = 0; i < blocks; ++i) {
        FreeNode* node = reinterpret_cast<FreeNode*>(base + i * m_blockSize);
        node->next = m_freeList;
        m_freeList = node;
    }
}

// Allocate memory.
void* PoolAllocator::allocate(std::size_t n) {
    // For simplicity, if n is not equal to m_blockSize, we ignore it and allocate one block.
    std::lock_guard<std::mutex> lock(m_mtx);
    if (!m_freeList) {
        allocateNewSlab();
    }
    // Pop a block from the free list.
    FreeNode* node = m_freeList;
    m_freeList = node->next;
    
    // Update slab bookkeeping.
    for (auto& slab : m_slabs) {
        char* start = static_cast<char*>(slab.memory);
        char* end = start + slab.capacity * m_blockSize;
        if (reinterpret_cast<char*>(node) >= start && reinterpret_cast<char*>(node) < end) {
            ++slab.used;
            break;
        }
    }
    return reinterpret_cast<void*>(node);
}

// Deallocate memory.
void PoolAllocator::deallocate(void* p, std::size_t n) {
    (void)n; // We assume n is m_blockSize.
    std::lock_guard<std::mutex> lock(m_mtx);
    if (!p)
        return;
    FreeNode* node = reinterpret_cast<FreeNode*>(p);
    node->next = m_freeList;
    m_freeList = node;
    
    // Update slab bookkeeping.
    for (auto& slab : m_slabs) {
        char* start = static_cast<char*>(slab.memory);
        char* end = start + slab.capacity * m_blockSize;
        if (reinterpret_cast<char*>(p) >= start && reinterpret_cast<char*>(p) < end) {
            assert(slab.used > 0);
            --slab.used;
            break;
        }
    }
    // Optionally, attempt to reclaim any entirely free slabs.
    tryReclaimSlabs();
}

// Try to reclaim (free) slabs that are entirely unused.
void PoolAllocator::tryReclaimSlabs() {
    // For simplicity, iterate through slabs and free those with slab.used == 0.
    // This is a simplistic approach; a production allocator may want to keep
    // a cache of free slabs.
    for (auto it = m_slabs.begin(); it != m_slabs.end(); ) {
        if (it->used == 0) {
            // Remove all blocks belonging to this slab from the free list.
            char* start = static_cast<char*>(it->memory);
            char* end = start + it->capacity * m_blockSize;
            FreeNode* prev = nullptr;
            FreeNode* curr = m_freeList;
            while (curr) {
                char* addr = reinterpret_cast<char*>(curr);
                if (addr >= start && addr < end) {
                    if (prev) {
                        prev->next = curr->next;
                    } else {
                        m_freeList = curr->next;
                    }
                    FreeNode* temp = curr;
                    curr = curr->next;
                    temp->next = nullptr; // not strictly necessary
                } else {
                    prev = curr;
                    curr = curr->next;
                }
            }
            // Free the slab memory.
#if defined(_MSC_VER)
            _aligned_free(it->memory);
#else
            free(it->memory);
#endif
            // Erase the slab.
            it = m_slabs.erase(it);
        } else {
            ++it;
        }
    }
}

// Return total capacity (number of blocks across all slabs).
std::size_t PoolAllocator::totalCapacity() const {
    std::size_t total = 0;
    for (const auto& slab : m_slabs) {
        total += slab.capacity;
    }
    return total;
}


