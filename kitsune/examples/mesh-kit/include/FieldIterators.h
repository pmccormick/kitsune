/**
 * @file FieldIterators.h
 * @brief Iterator classes for efficient traversal of Field data
 * 
 * This file defines a set of iterator classes for traversing Field data
 * in different ways (linear, 2D, blocked), with an emphasis on performance
 * and compatibility with both direct access and higher-level abstractions.
 */

#ifndef FIELD_ITERATORS_H
#define FIELD_ITERATORS_H

#include <iterator>
#include <cstddef>
#include <utility>

// Forward declarations
template <typename T, typename LocationTag, size_t D> class Field;

namespace FieldIterators {

/**
 * @brief Linear iterator for direct traversal of field data
 * 
 * Provides efficient linear traversal of the underlying field data
 * with minimal overhead. This is the most performant iterator for
 * bulk operations that don't need position information.
 * 
 * @tparam T Field data type
 * @tparam LocationTag Field location tag
 * @tparam D Field depth (default=1)
 */
template <typename T, typename LocationTag, size_t D = 1>
class LinearIterator {
public:
    // STL iterator type traits
    using iterator_category = std::random_access_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using reference = T&;
    
    /**
     * @brief Construct a new Linear Iterator
     * 
     * @param data Pointer to the field data
     * @param index Current index position (default=0)
     */
    LinearIterator(T* data, size_t index = 0) : m_data(data), m_index(index) {}
    
    // Core iterator operations
    reference operator*() const { return m_data[m_index]; }
    pointer operator->() const { return &m_data[m_index]; }
    
    // Increment/decrement
    LinearIterator& operator++() { ++m_index; return *this; }
    LinearIterator operator++(int) { auto tmp = *this; ++m_index; return tmp; }
    LinearIterator& operator--() { --m_index; return *this; }
    LinearIterator operator--(int) { auto tmp = *this; --m_index; return tmp; }
    
    // Random access
    LinearIterator& operator+=(difference_type n) { m_index += n; return *this; }
    LinearIterator operator+(difference_type n) const { return LinearIterator(m_data, m_index + n); }
    friend LinearIterator operator+(difference_type n, const LinearIterator& it) { return it + n; }
    
    LinearIterator& operator-=(difference_type n) { m_index -= n; return *this; }
    LinearIterator operator-(difference_type n) const { return LinearIterator(m_data, m_index - n); }
    difference_type operator-(const LinearIterator& other) const { return m_index - other.m_index; }
    
    // Comparison
    bool operator==(const LinearIterator& other) const { return m_index == other.m_index; }
    bool operator!=(const LinearIterator& other) const { return m_index != other.m_index; }
    bool operator<(const LinearIterator& other) const { return m_index < other.m_index; }
    bool operator<=(const LinearIterator& other) const { return m_index <= other.m_index; }
    bool operator>(const LinearIterator& other) const { return m_index > other.m_index; }
    bool operator>=(const LinearIterator& other) const { return m_index >= other.m_index; }
    
    // Subscript
    reference operator[](difference_type n) const { return m_data[m_index + n]; }
    
    // Current index
    size_t index() const { return m_index; }

private:
    T* m_data;        // Pointer to field data array
    size_t m_index;   // Current index
};

/**
 * @brief 2D iterator for traversing field data with position information
 * 
 * Provides traversal of field data while maintaining (i,j) position information.
 * This iterator is designed for algorithms that need to work with the 2D
 * structure of the field.
 * 
 * @tparam T Field data type
 * @tparam LocationTag Field location tag
 * @tparam D Field depth (default=1)
 */
  template <typename T, typename LocationTag, size_t D = 1>
  class Iterator2D {
  public:
    // STL iterator type traits (forward iterator only)
    using iterator_category = std::forward_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using reference = T&;
    
    /**
     * @brief Construct a new 2D Iterator
     * 
     * @param field Reference to the field
     * @param i Initial i-index (default=0)
     * @param j Initial j-index (default=0)
     * @param k Initial k-index (default=0)
     */
    Iterator2D(Field<T, LocationTag, D>& field, size_t i = 0, size_t j = 0, size_t k = 0)
      : m_field(field), m_i(i), m_j(j), m_k(k) {}
    
    // Core iterator operations
    reference operator*() { return m_field(m_i, m_j, m_k); }
    pointer operator->() { return &m_field(m_i, m_j, m_k); }
    
    // Increment (row-major order)
    Iterator2D& operator++() {
      ++m_k;  // First increment component index
        
      if (m_k >= D) {  // If we've gone through all components for this cell
	m_k = 0;     // Reset k to 0
	++m_i;       // Move to next cell in row
            
	if (m_i >= m_field.nx()) {  // If we've reached the end of a row
	  m_i = 0;                // Reset i to 0
	  ++m_j;                  // Move to next row
	}
      }
        
      return *this;
    }
    
    Iterator2D operator++(int) {
      Iterator2D tmp = *this;
      ++(*this);
      return tmp;
    }
    
    // Comparison
    bool operator==(const Iterator2D& other) const {
      return m_i == other.m_i && m_j == other.m_j && m_k == other.m_k;
    }
    
    bool operator!=(const Iterator2D& other) const {
      return !(*this == other);
    }
    
    // Access to current indices
    size_t i() const { return m_i; }
    size_t j() const { return m_j; }
    size_t k() const { return m_k; }
    
    // Get linear index
    size_t linearIndex() const { return m_field.index(m_i, m_j, m_k); }

  private:
    Field<T, LocationTag, D>& m_field;  // Reference to field
    size_t m_i, m_j, m_k;              // Current indices
  };

/**
 * @brief Range class for 2D iteration
 * 
 * Provides a convenient interface for range-based for loops with 2D iterators.
 * 
 * @tparam T Field data type
 * @tparam LocationTag Field location tag
 * @tparam D Field depth (default=1)
 */
template <typename T, typename LocationTag, size_t D = 1>
class Range2D {
public:
    using iterator = Iterator2D<T, LocationTag, D>;
    
    /**
     * @brief Construct a new 2D Range
     * 
     * @param field Reference to the field
     */
    Range2D(Field<T, LocationTag, D>& field) : m_field(field) {}
    
    /**
     * @brief Get iterator to beginning of range
     */
    iterator begin() { return iterator(m_field); }
    
    /**
     * @brief Get iterator to end of range
     */
    iterator end() {
        if constexpr (D > 1) {
            return iterator(m_field, 0, 0, m_field.depth());
        } else {
            return iterator(m_field, 0, m_field.ny());
        }
    }

private:
    Field<T, LocationTag, D>& m_field;
};

/**
 * @brief Block-based iterator for cache-efficient traversal
 * 
 * Provides traversal of field data in a blocked pattern for better
 * cache utilization. This is particularly useful for stencil operations
 * and other algorithms that benefit from spatial locality.
 * 
 * @tparam T Field data type
 * @tparam LocationTag Field location tag
 * @tparam D Field depth (default=1)
 */
template <typename T, typename LocationTag, size_t D = 1>
class BlockIterator {
public:
    // STL iterator type traits
    using iterator_category = std::forward_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using reference = T&;
    
    /**
     * @brief Construct a new Block Iterator
     * 
     * @param field Reference to the field
     * @param blockSizeX Block width (default=8)
     * @param blockSizeY Block height (default=8)
     * @param blockX Initial block x-index (default=0)
     * @param blockY Initial block y-index (default=0)
     * @param localI Initial local i-index within block (default=0)
     * @param localJ Initial local j-index within block (default=0)
     */
    BlockIterator(Field<T, LocationTag, D>& field, 
                 size_t blockSizeX = 8, size_t blockSizeY = 8,
                 size_t blockX = 0, size_t blockY = 0,
                 size_t localI = 0, size_t localJ = 0)
        : m_field(field), 
          m_blockSizeX(blockSizeX), m_blockSizeY(blockSizeY),
          m_blockX(blockX), m_blockY(blockY),
          m_localI(localI), m_localJ(localJ),
          m_numBlocksX((field.nx() + blockSizeX - 1) / blockSizeX),
          m_numBlocksY((field.ny() + blockSizeY - 1) / blockSizeY) {}
    
    // Core iterator operations
    reference operator*() {
        size_t i = m_blockX * m_blockSizeX + m_localI;
        size_t j = m_blockY * m_blockSizeY + m_localJ;
        return m_field(i, j);
    }
    
    pointer operator->() {
        size_t i = m_blockX * m_blockSizeX + m_localI;
        size_t j = m_blockY * m_blockSizeY + m_localJ;
        return &m_field(i, j);
    }
    
    // Increment (block-wise traversal)
    BlockIterator& operator++() {
        ++m_localI;
        
        // If we reach the end of a block row or the field boundary
        if (m_localI >= m_blockSizeX || 
            m_blockX * m_blockSizeX + m_localI >= m_field.nx()) {
            m_localI = 0;
            ++m_localJ;
            
            // If we reach the end of a block or the field boundary
            if (m_localJ >= m_blockSizeY || 
                m_blockY * m_blockSizeY + m_localJ >= m_field.ny()) {
                m_localJ = 0;
                ++m_blockX;
                
                // If we reach the end of a row of blocks
                if (m_blockX >= m_numBlocksX) {
                    m_blockX = 0;
                    ++m_blockY;
                }
            }
        }
        
        return *this;
    }
    
    BlockIterator operator++(int) {
        BlockIterator tmp = *this;
        ++(*this);
        return tmp;
    }
    
    // Comparison
    bool operator==(const BlockIterator& other) const {
        return m_blockX == other.m_blockX && 
               m_blockY == other.m_blockY &&
               m_localI == other.m_localI && 
               m_localJ == other.m_localJ;
    }
    
    bool operator!=(const BlockIterator& other) const {
        return !(*this == other);
    }
    
    // Access to current indices
    size_t i() const { return m_blockX * m_blockSizeX + m_localI; }
    size_t j() const { return m_blockY * m_blockSizeY + m_localJ; }
    
    // Check if iterator is at a valid position
    bool isValid() const {
        return m_blockY < m_numBlocksY && 
               i() < m_field.nx() && 
               j() < m_field.ny();
    }

private:
    Field<T, LocationTag, D>& m_field;  // Reference to field
    size_t m_blockSizeX, m_blockSizeY;  // Block dimensions
    size_t m_blockX, m_blockY;          // Current block indices
    size_t m_localI, m_localJ;          // Local indices within block
    size_t m_numBlocksX, m_numBlocksY;  // Number of blocks in each dimension
};

/**
 * @brief Range class for block-based iteration
 * 
 * Provides a convenient interface for range-based for loops with block iterators.
 * 
 * @tparam T Field data type
 * @tparam LocationTag Field location tag
 * @tparam D Field depth (default=1)
 */
template <typename T, typename LocationTag, size_t D = 1>
class BlockRange {
public:
    using iterator = BlockIterator<T, LocationTag, D>;
    
    /**
     * @brief Construct a new Block Range
     * 
     * @param field Reference to the field
     * @param blockSizeX Block width (default=8)
     * @param blockSizeY Block height (default=8)
     */
    BlockRange(Field<T, LocationTag, D>& field, size_t blockSizeX = 8, size_t blockSizeY = 8) 
        : m_field(field), m_blockSizeX(blockSizeX), m_blockSizeY(blockSizeY) {}
    
    /**
     * @brief Get iterator to beginning of range
     */
    iterator begin() { return iterator(m_field, m_blockSizeX, m_blockSizeY); }
    
    /**
     * @brief Get iterator to end of range
     */
    iterator end() {
        size_t numBlocksY = (m_field.ny() + m_blockSizeY - 1) / m_blockSizeY;
        return iterator(m_field, m_blockSizeX, m_blockSizeY, 0, numBlocksY);
    }

private:
    Field<T, LocationTag, D>& m_field;
    size_t m_blockSizeX, m_blockSizeY;
};

/**
 * @brief Strided iterator for parallel processing
 * 
 * Provides traversal of field data with a stride, allowing for easy
 * partitioning of work among multiple threads. This iterator is designed
 * for use in parallel processing contexts.
 * 
 * @tparam T Field data type
 * @tparam LocationTag Field location tag
 * @tparam D Field depth (default=1)
 */
template <typename T, typename LocationTag, size_t D = 1>
class StridedIterator {
public:
    // STL iterator type traits
    using iterator_category = std::forward_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using reference = T&;
    
    /**
     * @brief Construct a new Strided Iterator
     * 
     * @param field Reference to the field
     * @param startIdx Starting linear index
     * @param stride Step size between elements
     * @param endIdx Ending linear index (exclusive)
     */
    StridedIterator(Field<T, LocationTag, D>& field, 
                   size_t startIdx = 0, 
                   size_t stride = 1,
                   size_t endIdx = 0)
        : m_field(field), 
          m_currentIdx(startIdx), 
          m_stride(stride),
          m_endIdx(endIdx == 0 ? field.size() : endIdx) {}
    
    // Core iterator operations
    reference operator*() {
        // Convert linear index to field indices and access
        size_t linearIdx = m_currentIdx;
        size_t stride = m_field.getXStride();
        size_t k = linearIdx % D;
        linearIdx /= D;
        size_t j = linearIdx / stride;
        size_t i = linearIdx % stride;
        return m_field(i, j, k);
    }
    
    pointer operator->() {
        return &(operator*());
    }
    
    // Increment (by stride)
    StridedIterator& operator++() {
        m_currentIdx += m_stride;
        return *this;
    }
    
    StridedIterator operator++(int) {
        StridedIterator tmp = *this;
        m_currentIdx += m_stride;
        return tmp;
    }
    
    // Comparison
    bool operator==(const StridedIterator& other) const {
        return m_currentIdx == other.m_currentIdx;
    }
    
    bool operator!=(const StridedIterator& other) const {
        return m_currentIdx != other.m_currentIdx;
    }
    
    // Check if iterator is at a valid position
    bool isValid() const {
        return m_currentIdx < m_endIdx;
    }
    
    // Get current linear index
    size_t linearIndex() const { return m_currentIdx; }

private:
    Field<T, LocationTag, D>& m_field;  // Reference to field
    size_t m_currentIdx;               // Current linear index
    size_t m_stride;                   // Step size
    size_t m_endIdx;                   // End index
};

/**
 * @brief Range class for strided iteration
 * 
 * Provides a convenient interface for range-based for loops with strided iterators.
 * This is particularly useful for parallel processing.
 * 
 * @tparam T Field data type
 * @tparam LocationTag Field location tag
 * @tparam D Field depth (default=1)
 */
  template <typename T, typename LocationTag, size_t D = 1>
  class StridedRange {
  public:
    using iterator = StridedIterator<T, LocationTag, D>;
    
    /**
     * @brief Construct a new Strided Range
     * 
     * @param field Reference to the field
     * @param startIdx Starting linear index
     * @param endIdx Ending linear index (exclusive)
     * @param stride Step size between elements
     */
    StridedRange(Field<T, LocationTag, D>& field, 
		 size_t startIdx, size_t endIdx, size_t stride = 1) 
      : m_field(field), m_startIdx(startIdx), m_endIdx(endIdx), m_stride(stride) {}
    
    /**
     * @brief Construct a range for a specific partition
     * 
     * Creates a range for a specific partition of the field data,
     * useful for dividing work among multiple threads.
     * 
     * @param field Reference to the field
     * @param partitionId Partition ID (0-based)
     * @param numPartitions Total number of partitions
     */
    StridedRange(Field<T, LocationTag, D>& field, size_t partitionId, size_t numPartitions) 
      : m_field(field) {
      size_t totalSize = field.size();
      size_t partitionSize = (totalSize + numPartitions - 1) / numPartitions;
        
      m_startIdx = partitionId * partitionSize;
      m_endIdx = std::min((partitionId + 1) * partitionSize, totalSize);
      m_stride = 1;
    }

    static StridedRange forPartition(Field<T, LocationTag, D>& field, 
				     size_t partitionId, size_t numPartitions) {
      size_t totalSize = field.size();
      size_t partitionSize = (totalSize + numPartitions - 1) / numPartitions;
        
      size_t startIdx = partitionId * partitionSize;
      size_t endIdx = std::min((partitionId + 1) * partitionSize, totalSize);
        
      return StridedRange(field, startIdx, endIdx, 1);
    }
    
    /**
     * @brief Get iterator to beginning of range
     */
    iterator begin() { return iterator(m_field, m_startIdx, m_stride, m_endIdx); }
    
    /**
     * @brief Get iterator to end of range
     */
    iterator end() { return iterator(m_field, m_endIdx, m_stride, m_endIdx); }

  

  private:
    Field<T, LocationTag, D>& m_field;
    size_t m_startIdx, m_endIdx, m_stride;
  };

} // namespace FieldIterators

#endif // FIELD_ITERATORS_H


