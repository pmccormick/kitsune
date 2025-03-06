/**
 * @file AlignedAllocator.h
 * @brief CPU allocator for aligned memory using custom allocation routines.
 *
 * ================================================================================
 * Design Considerations:
 * ================================================================================
 * This allocator is designed to provide memory that is aligned to a given boundary,
 * which is essential for achieving high performance on modern CPUs that rely on SIMD
 * instructions and cache line efficiency. By default, we set the alignment to 64 bytes,
 * but this is configurable via a template parameter.
 *
 * Pros:
 *   - Guarantees contiguous memory that is properly aligned for vectorization.
 *   - Can significantly improve performance in computational kernels that use SIMD.
 *   - Transparent integration with std::vector via custom allocator interface.
 *
 * Cons:
 *   - Requires platform–specific code. This implementation uses posix_memalign
 *     on POSIX systems and _aligned_malloc on Windows.
 *   - Slight overhead in allocation/deallocation compared to default malloc/free,
 *     though this is usually outweighed by the performance benefits in compute–intensive code.
 *
 * Future Considerations:
 *   - Integration with high–performance memory libraries.
 *   - Optionally supporting allocator tracking and debugging.
 *
 * ================================================================================
 */

#ifndef ALIGNED_ALLOCATOR_H
#define ALIGNED_ALLOCATOR_H

#include <cstdlib>
#include <cstddef>
#include <stdexcept>
#include <limits>
#include <memory>
#include <new>

// Cross-platform support: use posix_memalign if available or _aligned_malloc on Windows.
#if defined(_MSC_VER)
  #include <malloc.h>
#endif

template<typename T, std::size_t Alignment = 64>
class AlignedAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    AlignedAllocator() noexcept {}
    template <typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    // Allocate memory for n objects of type T with specified alignment.
    pointer allocate(size_type n, const void* hint = 0) {
        (void)hint;
        if (n > std::numeric_limits<size_type>::max() / sizeof(T))
            throw std::bad_alloc();

        void* ptr = nullptr;
#if defined(_MSC_VER)
        ptr = _aligned_malloc(n * sizeof(T), Alignment);
        if (!ptr)
            throw std::bad_alloc();
#else
        if (posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0)
            throw std::bad_alloc();
#endif
        return static_cast<pointer>(ptr);
    }

    // Deallocate memory.
    void deallocate(pointer p, size_type n) noexcept {
        (void)n;
#if defined(_MSC_VER)
        _aligned_free(p);
#else
        free(p);
#endif
    }

    // Equality operators.
    template <typename U, std::size_t OtherAlignment>
    bool operator==(const AlignedAllocator<U, OtherAlignment>&) const noexcept {
        return Alignment == OtherAlignment;
    }
    template <typename U, std::size_t OtherAlignment>
    bool operator!=(const AlignedAllocator<U, OtherAlignment>& other) const noexcept {
        return !(*this == other);
    }
};

#endif // ALIGNED_ALLOCATOR_H


