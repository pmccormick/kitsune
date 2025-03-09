/**
 * @file Field.h
 * @brief Templated Field class for high-performance data storage
 *
 * This implementation uses templated classes for high performance,
 * making it suitable for use with code generation where field types are known
 * at generation time. This approach provides better performance and cleaner access
 * syntax with minimal runtime overhead.
 * 
 * Field Access Strategy:
 * ----------------------
 * The Field class is designed to work within the code generation framework where:
 * 1. Specialized Mesh classes store fields as member variables
 * 2. Generated Cell subclasses access these fields directly
 * 3. Optimized memory layouts and access patterns are used for performance
 * 
 * The Field class is intentionally kept lightweight with core functionality only.
 * Operations like gradients, divergence, curl, etc. are implemented as standalone
 * functions in separate files.
 */

#ifndef FIELD_H
#define FIELD_H

#include "FieldIterators.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

//------------------------------------------------------------------------------
// Type tag structures to denote where a field's data is stored in a mesh grid.
// These tags are used solely for compile-time type safety and clarity.
struct CellCenterTag {};
struct VertexTag {};
struct HorizontalEdgeTag {};
struct VerticalEdgeTag {};

//------------------------------------------------------------------------------
// Field class template.
//   T: the data type (for example, double)
//   LocationTag: a type tag that indicates the storage location.
//                Use one of: CellCenterTag, VertexTag, HorizontalEdgeTag,
//                VerticalEdgeTag.
//   D: the third dimension size (default=1 for 2D fields)
template <typename T, typename LocationTag, size_t D = 1> 
class Field {
public:
  // Type definitions for STL compatibility
  using value_type = T;
  using reference = T&;
  using const_reference = const T&;
  using pointer = T*;
  using const_pointer = const T*;
  using iterator = typename std::vector<T>::iterator;
  using const_iterator = typename std::vector<T>::const_iterator;
  using size_type = size_t;
  using difference_type = std::ptrdiff_t;

  /**
   * @brief Constructor.
   *
   * @param nx Base number of cells in x-direction (for cell-centered data).
   * @param ny Base number of cells in y-direction (for cell-centered data).
   *
   * The actual dimensions for storage are determined by the LocationTag:
   *   - CellCenterTag:      Dimensions: nx x ny x D
   *   - VertexTag:          Dimensions: (nx+1) x (ny+1) x D
   *   - HorizontalEdgeTag:  Dimensions: nx x (ny+1) x D
   *   - VerticalEdgeTag:    Dimensions: (nx+1) x ny x D
   */
  Field(size_t nx, size_t ny) : m_nx(nx), m_ny(ny) {
    size_t total = computeTotalSize();
    m_data.resize(total);
  }

  /**
   * @brief Copy constructor
   */
  Field(const Field& other) = default;

  /**
   * @brief Move constructor
   */
  Field(Field&& other) noexcept = default;

  /**
   * @brief Copy assignment operator
   */
  Field& operator=(const Field& other) = default;

  /**
   * @brief Move assignment operator
   */
  Field& operator=(Field&& other) noexcept = default;

  /**
   * @brief Overloaded access operator for 2D fields (non-const).
   *
   * Provides inline 2D access to the field data.
   *
   * @param i Index in the x-direction.
   * @param j Index in the y-direction.
   * @return Reference to the element at (i,j).
   */
  inline T& operator()(size_t i, size_t j) {
    size_t idx = index(i, j, 0);
    assert(idx < m_data.size());
    return m_data[idx];
  }

  /**
   * @brief Overloaded access operator for 2D fields (const).
   *
   * @param i Index in the x-direction.
   * @param j Index in the y-direction.
   * @return Const reference to the element at (i,j).
   */
  inline const T& operator()(size_t i, size_t j) const {
    size_t idx = index(i, j, 0);
    assert(idx < m_data.size());
    return m_data[idx];
  }

  /**
   * @brief Overloaded access operator for 3D fields (non-const).
   *
   * Provides inline 3D access to the field data.
   *
   * @param i Index in the x-direction.
   * @param j Index in the y-direction.
   * @param k Index in the third dimension.
   * @return Reference to the element at (i,j,k).
   */
  inline T& operator()(size_t i, size_t j, size_t k) {
    size_t idx = index(i, j, k);
    assert(idx < m_data.size());
    return m_data[idx];
  }

  /**
   * @brief Overloaded access operator for 3D fields (const).
   *
   * @param i Index in the x-direction.
   * @param j Index in the y-direction.
   * @param k Index in the third dimension.
   * @return Const reference to the element at (i,j,k).
   */
  inline const T& operator()(size_t i, size_t j, size_t k) const {
    size_t idx = index(i, j, k);
    assert(idx < m_data.size());
    return m_data[idx];
  }

  /**
   * @brief Direct linear array access (non-const)
   * 
   * @param idx Linear index
   * @return Reference to the element
   */
  inline T& operator[](size_t idx) {
    assert(idx < m_data.size());
    return m_data[idx];
  }

  /**
   * @brief Direct linear array access (const)
   * 
   * @param idx Linear index
   * @return Const reference to the element
   */
  inline const T& operator[](size_t idx) const {
    assert(idx < m_data.size());
    return m_data[idx];
  }

  // Iterator support for STL algorithms
  iterator begin() { return m_data.begin(); }
  iterator end() { return m_data.end(); }
  const_iterator begin() const { return m_data.begin(); }
  const_iterator end() const { return m_data.end(); }
  const_iterator cbegin() const { return m_data.cbegin(); }
  const_iterator cend() const { return m_data.cend(); }

  /// @brief Returns the total number of elements in the field.
  inline size_t size() const { return m_data.size(); }

  /// @brief Returns the size in the x-direction.
  inline size_t nx() const {
    if constexpr (std::is_same_v<LocationTag, CellCenterTag> || 
		  std::is_same_v<LocationTag, HorizontalEdgeTag>) {
      return m_nx;
    } else { // vertex or vertical edge
      return m_nx + 1;
    }
  }

  /// @brief Returns the size in the y-direction.
  inline size_t ny() const {
    if constexpr (std::is_same_v<LocationTag, CellCenterTag> || 
		  std::is_same_v<LocationTag, VerticalEdgeTag>) {
      return m_ny;
    } else { // vertex or horizontal edge
      return m_ny + 1;
    }
  }

  /// @brief Returns the size in the third dimension.
  inline constexpr size_t depth() const { return D; }

  /// @brief Returns the base number of cells in x-direction.
  inline size_t baseNx() const { return m_nx; }

  /// @brief Returns the base number of cells in y-direction.
  inline size_t baseNy() const { return m_ny; }

  /**
   * @brief Computes the X stride in memory layout
   * 
   * @return The stride in the X dimension
   */
  inline constexpr size_t getXStride() const {
    if constexpr (std::is_same_v<LocationTag, CellCenterTag> || 
		  std::is_same_v<LocationTag, HorizontalEdgeTag>) {
      return m_nx;
    } else { // vertex or vertical edge
      return m_nx + 1;
    }
  }

  /**
   * @brief Computes the one-dimensional index from three-dimensional (i,j,k)
   * indices.
   *
   * The valid range for (i,j) depends on the field type:
   *   - For CellCenterTag:      i in [0, nx),     j in [0, ny),     k in [0, D)
   *   - For VertexTag:          i in [0, nx+1),   j in [0, ny+1),   k in [0, D)
   *   - For HorizontalEdgeTag:  i in [0, nx),     j in [0, ny+1),   k in [0, D)
   *   - For VerticalEdgeTag:    i in [0, nx+1),   j in [0, ny),     k in [0, D)
   *
   * @param i Index in x-direction.
   * @param j Index in y-direction.
   * @param k Index in third dimension.
   * @return Linear index into the data vector.
   */
  inline size_t index(size_t i, size_t j, size_t k) const {
    const size_t stride = getXStride();
    return (i + j * stride) * D + k;
  }

  /**
   * @brief Fill the entire field with a value.
   * @param value The value to fill with.
   */
  void fill(const T& value) { 
    std::fill(m_data.begin(), m_data.end(), value); 
  }

  /**
   * @brief Fill the field with zeros.
   */
  void fillZero() {
    if constexpr (std::is_trivially_constructible_v<T> && 
		  (std::is_integral_v<T> || std::is_floating_point_v<T>)) {
      // For trivial types, use memset for better performance
      std::memset(m_data.data(), 0, m_data.size() * sizeof(T));
    } else {
      // For non-trivial types, use std::fill
      std::fill(m_data.begin(), m_data.end(), T{});
    }
  }

  /**
   * @brief Get direct access to the underlying data vector.
   * @return Reference to the data vector.
   */
  std::vector<T>& data() { return m_data; }

  /**
   * @brief Get const access to the underlying data vector.
   * @return Const reference to the data vector.
   */
  const std::vector<T>& data() const { return m_data; }

  /**
   * @brief Get pointer to the raw data array
   * @return Pointer to the first element of the data array
   */
  T* rawData() { return m_data.data(); }
    
  /**
   * @brief Get const pointer to the raw data array
   * @return Const pointer to the first element of the data array
   */
  const T* rawData() const { return m_data.data(); }

  /**
   * @brief Add another field to this field
   * 
   * @param other Field to add
   * @return Reference to this field
   */
  Field& operator+=(const Field& other) {
    assert(size() == other.size());
    for (size_t i = 0; i < size(); ++i) {
      m_data[i] += other.m_data[i];
    }
    return *this;
  }

  /**
   * @brief Subtract another field from this field
   * 
   * @param other Field to subtract
   * @return Reference to this field
   */
  Field& operator-=(const Field& other) {
    assert(size() == other.size());
    for (size_t i = 0; i < size(); ++i) {
      m_data[i] -= other.m_data[i];
    }
    return *this;
  }

  /**
   * @brief Multiply this field by another field element-wise
   * 
   * @param other Field to multiply by
   * @return Reference to this field
   */
  Field& operator*=(const Field& other) {
    assert(size() == other.size());
    for (size_t i = 0; i < size(); ++i) {
      m_data[i] *= other.m_data[i];
    }
    return *this;
  }

  /**
   * @brief Divide this field by another field element-wise
   * 
   * @param other Field to divide by
   * @return Reference to this field
   */
  Field& operator/=(const Field& other) {
    assert(size() == other.size());
    for (size_t i = 0; i < size(); ++i) {
      m_data[i] /= other.m_data[i];
    }
    return *this;
  }

  /**
   * @brief Scale the field by a constant
   * 
   * @param scalar The scalar value to multiply by
   * @return Reference to this field
   */
  Field& operator*=(const T& scalar) {
    for (size_t i = 0; i < size(); ++i) {
      m_data[i] *= scalar;
    }
    return *this;
  }

  /**
   * @brief Divide the field by a constant
   * 
   * @param scalar The scalar value to divide by
   * @return Reference to this field
   */
  Field& operator/=(const T& scalar) {
    T invScalar = static_cast<T>(1) / scalar;
    return (*this) *= invScalar;
  }

  /**
   * @brief Add a constant to all elements
   * 
   * @param scalar The scalar value to add
   * @return Reference to this field
   */
  Field& operator+=(const T& scalar) {
    for (size_t i = 0; i < size(); ++i) {
      m_data[i] += scalar;
    }
    return *this;
  }

  /**
   * @brief Subtract a constant from all elements
   * 
   * @param scalar The scalar value to subtract
   * @return Reference to this field
   */
  Field& operator-=(const T& scalar) {
    for (size_t i = 0; i < size(); ++i) {
      m_data[i] -= scalar;
    }
    return *this;
  }

  /**
   * @brief Apply an arbitrary function to all elements
   * 
   * @param func The function to apply
   */
  template <typename Func>
  void forEach(Func func) {
    for (auto& val : m_data) {
      val = func(val);
    }
  }

  /**
   * @brief Apply a function to each element with its indices
   * 
   * @param func Function to apply (taking i, j, and value reference)
   */
  template <typename Func>
  void forEachIndexed(Func func) {
    const size_t nx_val = nx();
    const size_t ny_val = ny();
        
    for (size_t j = 0; j < ny_val; ++j) {
      for (size_t i = 0; i < nx_val; ++i) {
	for (size_t k = 0; k < D; ++k) {
	  func(i, j, k, (*this)(i, j, k));
	}
      }
    }
  }

  /**
   * @brief Calculate the minimum value in the field
   * 
   * @return The minimum value
   */
  T min() const {
    if (m_data.empty()) return T();
    return *std::min_element(m_data.begin(), m_data.end());
  }

  /**
   * @brief Calculate the maximum value in the field
   * 
   * @return The maximum value
   */
  T max() const {
    if (m_data.empty()) return T();
    return *std::max_element(m_data.begin(), m_data.end());
  }

  /**
   * @brief Calculate the sum of all values in the field
   * 
   * @return The sum
   */
  T sum() const {
    T result = T();
    for (const auto& val : m_data) {
      result += val;
    }
    return result;
  }

  /**
   * @brief Calculate the average of all values in the field
   * 
   * @return The average
   */
  double average() const {
    if (m_data.empty()) return 0.0;
    return static_cast<double>(sum()) / m_data.size();
  }

  /**
   * @brief Swap contents with another field
   * 
   * @param other The field to swap with
   */
  void swap(Field& other) noexcept {
    std::swap(m_nx, other.m_nx);
    std::swap(m_ny, other.m_ny);
    m_data.swap(other.m_data);
  }

  /**
   * @brief Resize the field
   * 
   * Resizes the field to the new dimensions, preserving data where possible.
   * Note that this may invalidate existing references to elements.
   * 
   * @param nx New base number of cells in x-direction
   * @param ny New base number of cells in y-direction
   */
  void resize(size_t nx, size_t ny) {
    if (nx == m_nx && ny == m_ny) return;
        
    m_nx = nx;
    m_ny = ny;
    m_data.resize(computeTotalSize());
  }

  /**
   * @brief Copy data from another field
   * 
   * @param other Source field to copy from
   */
  void copyFrom(const Field& other) {
    if (nx() != other.nx() || ny() != other.ny() || depth() != other.depth()) {
      resize(other.baseNx(), other.baseNy());
    }
        
    std::copy(other.m_data.begin(), other.m_data.end(), m_data.begin());
  }
  // Iterator type aliases
  using LinearIterator = FieldIterators::LinearIterator<T, LocationTag, D>;
  using Iterator2D = FieldIterators::Iterator2D<T, LocationTag, D>;
  using BlockIterator = FieldIterators::BlockIterator<T, LocationTag, D>;
  using StridedIterator = FieldIterators::StridedIterator<T, LocationTag, D>;

  // Range type aliases
  using Range2D = FieldIterators::Range2D<T, LocationTag, D>;
  using BlockRange = FieldIterators::BlockRange<T, LocationTag, D>;
  using StridedRange = FieldIterators::StridedRange<T, LocationTag, D>;

  // Linear iterator methods (complementing existing STL iterators)
  LinearIterator linearBegin() { return LinearIterator(m_data.data()); }
  LinearIterator linearEnd() { return LinearIterator(m_data.data(), m_data.size()); }

  // 2D iterator methods
  Iterator2D begin2D() { return Iterator2D(*this); }
  Iterator2D end2D() {
    if constexpr (D > 1) {
      return Iterator2D(*this, 0, 0, D);
    } else {
      return Iterator2D(*this, 0, ny());
    }
  }

  // 2D range method
  Range2D range2D() { return Range2D(*this); }

  // Block iterator methods
  BlockIterator beginBlock(size_t blockSizeX = 8, size_t blockSizeY = 8) {
    return BlockIterator(*this, blockSizeX, blockSizeY);
  }

  BlockIterator endBlock(size_t blockSizeX = 8, size_t blockSizeY = 8) {
    size_t numBlocksY = (ny() + blockSizeY - 1) / blockSizeY;
    return BlockIterator(*this, blockSizeX, blockSizeY, 0, numBlocksY);
  }

  // Block range method
  BlockRange blockRange(size_t blockSizeX = 8, size_t blockSizeY = 8) {
    return BlockRange(*this, blockSizeX, blockSizeY);
  }

  // Strided iterator methods (for future parallel processing)
  StridedRange getPartition(size_t partitionId, size_t numPartitions) {
    return StridedRange::forPartition(*this, partitionId, numPartitions);
  }

  // Custom range method for a specific region
  template <typename Predicate>
  auto whereIndices(Predicate predicate) {
    std::vector<std::pair<size_t, size_t>> indices;
    for (size_t j = 0; j < ny(); ++j) {
      for (size_t i = 0; i < nx(); ++i) {
	if (predicate(i, j)) {
	  indices.emplace_back(i, j);
	}
      }
    }

    class IndexIterator {
    public:
      IndexIterator(Field& field, const std::vector<std::pair<size_t, size_t>>& indices, size_t pos = 0)
	: m_field(field), m_indices(indices), m_pos(pos) {}

      T& operator*() {
	auto [i, j] = m_indices[m_pos];
	return m_field(i, j);
      }

      IndexIterator& operator++() { ++m_pos; return *this; }
      bool operator!=(const IndexIterator& other) const { return m_pos != other.m_pos; }

      size_t i() const { return m_indices[m_pos].first; }
      size_t j() const { return m_indices[m_pos].second; }

    private:
      Field& m_field;
      const std::vector<std::pair<size_t, size_t>>& m_indices;
      size_t m_pos;
    };

    class IndexRange {
    public:
      IndexRange(Field& field, std::vector<std::pair<size_t, size_t>> indices)
	: m_field(field), m_indices(std::move(indices)) {}

      IndexIterator begin() { return IndexIterator(m_field, m_indices); }
      IndexIterator end() { return IndexIterator(m_field, m_indices, m_indices.size()); }

      size_t size() const { return m_indices.size(); }

    private:
      Field& m_field;
      std::vector<std::pair<size_t, size_t>> m_indices;
    };

    return IndexRange(*this, std::move(indices));
  }

private:
  size_t m_nx;           // Base number of cells in x-direction.
  size_t m_ny;           // Base number of cells in y-direction.
  std::vector<T> m_data; // Contiguous storage for field data.

  /**
   * @brief Computes the total number of elements to allocate.
   *
   * The size depends on the field type (LocationTag):
   *   - For CellCenterTag:      nx * ny * D
   *   - For VertexTag:          (nx+1) * (ny+1) * D
   *   - For HorizontalEdgeTag:  nx * (ny+1) * D
   *   - For VerticalEdgeTag:    (nx+1) * ny * D
   *
   * @return Total number of elements.
   */
  inline size_t computeTotalSize() const {
    const size_t x_size = nx();
    const size_t y_size = ny();
    return x_size * y_size * D;
  }
};

//------------------------------------------------------------------------------
// Convenience type definitions for common field types. These aliases make code
// more self-documenting and reduce the need to explicitly write the type tag.
template <typename T = double> using CellCenterField = Field<T, CellCenterTag>;

template <typename T = double> using VertexField = Field<T, VertexTag>;

template <typename T = double>
using HorizontalEdgeField = Field<T, HorizontalEdgeTag>;

template <typename T = double>
using VerticalEdgeField = Field<T, VerticalEdgeTag>;

// Vector fields for specific purposes (e.g., velocity, momentum)
template <typename T = double>
using VectorField = Field<T, CellCenterTag, 2>;

template <typename T = double>
using Vector3DField = Field<T, CellCenterTag, 3>;

// Binary operators for Field + Field
template <typename T, typename LocationTag, size_t D>
Field<T, LocationTag, D> operator+(const Field<T, LocationTag, D>& lhs, const Field<T, LocationTag, D>& rhs) {
    Field<T, LocationTag, D> result(lhs);
    result += rhs;
    return result;
}

template <typename T, typename LocationTag, size_t D>
Field<T, LocationTag, D> operator-(const Field<T, LocationTag, D>& lhs, const Field<T, LocationTag, D>& rhs) {
    Field<T, LocationTag, D> result(lhs);
    result -= rhs;
    return result;
}

template <typename T, typename LocationTag, size_t D>
Field<T, LocationTag, D> operator*(const Field<T, LocationTag, D>& lhs, const Field<T, LocationTag, D>& rhs) {
    Field<T, LocationTag, D> result(lhs);
    result *= rhs;
    return result;
}

template <typename T, typename LocationTag, size_t D>
Field<T, LocationTag, D> operator/(const Field<T, LocationTag, D>& lhs, const Field<T, LocationTag, D>& rhs) {
    Field<T, LocationTag, D> result(lhs);
    result /= rhs;
    return result;
}

// Binary operators for Field + Scalar
template <typename T, typename LocationTag, size_t D>
Field<T, LocationTag, D> operator+(const Field<T, LocationTag, D>& field, const T& scalar) {
    Field<T, LocationTag, D> result(field);
    result += scalar;
    return result;
}

template <typename T, typename LocationTag, size_t D>
Field<T, LocationTag, D> operator-(const Field<T, LocationTag, D>& field, const T& scalar) {
    Field<T, LocationTag, D> result(field);
    result -= scalar;
    return result;
}

template <typename T, typename LocationTag, size_t D>
Field<T, LocationTag, D> operator*(const Field<T, LocationTag, D>& field, const T& scalar) {
    Field<T, LocationTag, D> result(field);
    result *= scalar;
    return result;
}

template <typename T, typename LocationTag, size_t D>
Field<T, LocationTag, D> operator/(const Field<T, LocationTag, D>& field, const T& scalar) {
    Field<T, LocationTag, D> result(field);
    result /= scalar;
    return result;
}

// Binary operators for Scalar + Field
template <typename T, typename LocationTag, size_t D>
Field<T, LocationTag, D> operator+(const T& scalar, const Field<T, LocationTag, D>& field) {
    Field<T, LocationTag, D> result(field);
    result += scalar;
    return result;
}

template <typename T, typename LocationTag, size_t D>
Field<T, LocationTag, D> operator*(const T& scalar, const Field<T, LocationTag, D>& field) {
    Field<T, LocationTag, D> result(field);
    result *= scalar;
    return result;
}

#endif // FIELD_H


