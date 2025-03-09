/**
 * @file AccessorIterators.h
 * @brief Iterator interfaces for Accessor classes
 * 
 * This file defines iterator interfaces for the various accessor classes
 * (MeshAccessor, RegionAccessor, FieldAccessor, CompoundAccessor).
 * These iterators build on the Field iterators to provide consistent and
 * efficient traversal patterns for mesh data.
 */

#ifndef MESH_ACCESSOR_ITERATORS_H
#define MESH_ACCESSOR_ITERATORS_H

#include "AccessorIteratorsCommon.h"
#include <functional>


template <typename MeshType, typename CellType>
class MeshAccessorIterators {
public:
  /**
   * @brief Iterator for all cells in the mesh
   */
  class CellIterator {
  public:
    // STL iterator type traits
    using iterator_category = std::forward_iterator_tag;
    using value_type = CellType*;
    using difference_type = std::ptrdiff_t;
    using pointer = CellType**;
    using reference = CellType*&;
        
    /**
     * @brief Construct a new Cell Iterator
     * 
     * @param mesh Reference to the mesh
     * @param i Initial i-index
     * @param j Initial j-index
     */
    CellIterator(MeshType& mesh, int i = 0, int j = 0)
      : m_mesh(mesh), m_i(i), m_j(j) {
      // Find first valid cell if necessary
      if (i == 0 && j == 0 && !m_mesh.getTypedCell(i, j)) {
	findNextValidCell();
      }
    }
        
    // Core iterator operations
    value_type operator*() const { return m_mesh.getTypedCell(m_i, m_j); }
        
    CellIterator& operator++() {
      ++m_i;
      if (m_i >= m_mesh.nx()) {
	m_i = 0;
	++m_j;
      }
            
      // Find next valid cell if necessary
      if (m_j < m_mesh.ny() && !m_mesh.getTypedCell(m_i, m_j)) {
	findNextValidCell();
      }
            
      return *this;
    }
        
    CellIterator operator++(int) {
      CellIterator tmp = *this;
      ++(*this);
      return tmp;
    }
        
    // Comparison
    bool operator==(const CellIterator& other) const {
      return m_i == other.m_i && m_j == other.m_j;
    }
        
    bool operator!=(const CellIterator& other) const {
      return !(*this == other);
    }
        
    // Current indices
    int i() const { return m_i; }
    int j() const { return m_j; }
        
  private:
    void findNextValidCell() {
      while (m_j < m_mesh.ny()) {
	if (m_mesh.getTypedCell(m_i, m_j)) {
	  return;
	}
                
	++m_i;
	if (m_i >= m_mesh.nx()) {
	  m_i = 0;
	  ++m_j;
	}
      }
    }
        
    MeshType& m_mesh;
    int m_i, m_j;
  };


  class BoundaryCellIterator {
  public:
    // STL iterator type traits
    using iterator_category = std::forward_iterator_tag;
    using value_type = CellType*;
    using difference_type = std::ptrdiff_t;
    using pointer = CellType**;
    using reference = CellType*&;
    
    /**
     * @brief Construct a new Boundary Cell Iterator
     * 
     * @param mesh Reference to the mesh
     * @param side Current boundary side (0=bottom, 1=right, 2=top, 3=left, 4=done)
     * @param pos Position along current side
     */
    BoundaryCellIterator(MeshType& mesh, int side = 0, int pos = 0)
      : m_mesh(mesh), m_side(side), m_pos(pos) {
      // Initialize with first cell or end state
      if (side == 0 && pos == 0) {
	m_current = m_mesh.getTypedCell(0, 0);
      } else if (side == 4) {
	m_current = nullptr;
      } else {
	updateCurrentCell();
      }
    }
    
    // Core iterator operations
    value_type operator*() const { return m_current; }
    
    BoundaryCellIterator& operator++() {
      ++m_pos;
        
      // Check if we've reached the end of the current side
      switch (m_side) {
      case 0: // Bottom row
	if (m_pos >= m_mesh.nx()) {
	  m_side = 1; // Move to right column
	  m_pos = 0;
	}
	break;
                
      case 1: // Right column 
	if (m_pos >= m_mesh.ny() - 2) { // Skip top-right corner
	  m_side = 2; // Move to top row
	  m_pos = 0;
	}
	break;
                
      case 2: // Top row
	if (m_pos >= m_mesh.nx() - 2) { // Skip top-left corner
	  m_side = 3; // Move to left column
	  m_pos = 0;
	}
	break;
                
      case 3: // Left column
	if (m_pos >= m_mesh.ny() - 2) { // Skip bottom-left corner (already visited)
	  m_side = 4; // Done
	  m_pos = 0;
	}
	break;
      }
        
      // Update current cell based on new position
      updateCurrentCell();
      return *this;
    }
    
    BoundaryCellIterator operator++(int) {
      BoundaryCellIterator tmp = *this;
      ++(*this);
      return tmp;
    }
    
    // Comparison
    bool operator==(const BoundaryCellIterator& other) const {
      if (m_side == 4 && other.m_side == 4) return true;
      return m_side == other.m_side && m_pos == other.m_pos;
    }
    
    bool operator!=(const BoundaryCellIterator& other) const {
      return !(*this == other);
    }
    
    // Current indices
    int i() const { 
      switch (m_side) {
      case 0: // Bottom row
	return m_pos;
      case 1: // Right column
	return m_mesh.nx() - 1;
      case 2: // Top row
	return m_mesh.nx() - 2 - m_pos;
      case 3: // Left column
	return 0;
      default:
	return 0;
      }
    }
    
    int j() const { 
      switch (m_side) {
      case 0: // Bottom row
	return 0;
      case 1: // Right column
	return 1 + m_pos; // Skip bottom-right corner
      case 2: // Top row
	return m_mesh.ny() - 1;
      case 3: // Left column
	return m_mesh.ny() - 2 - m_pos; // Skip top-left corner
      default:
	return 0;
      }
    }
    
  private:
    void updateCurrentCell() {
      if (m_side < 4) {
	m_current = m_mesh.getTypedCell(i(), j());
      } else {
	m_current = nullptr;
      }
    }
    
    MeshType& m_mesh;
    int m_side;  // 0=bottom, 1=right, 2=top, 3=left, 4=done
    int m_pos;   // Position along current side
    CellType* m_current;
  };

  /**
 * @brief Iterator for interior cells only
 * 
 * This iterator traverses only the interior cells of a mesh, which are cells that are
 * not on the boundary. Specifically, interior cells satisfy:
 *   i > 0 && i < nx-1 && j > 0 && j < ny-1
 * 
 * Special cases:
 * - For meshes with nx <= 2 or ny <= 2, there are no interior cells.
 *   In these cases, begin() == end() immediately.
 * - If a mesh has "holes" (null cells) in its interior, these are skipped.
 */
class InteriorCellIterator {
public:
    // STL iterator type traits
    using iterator_category = std::forward_iterator_tag;
    using value_type = CellType*;
    using difference_type = std::ptrdiff_t;
    using pointer = CellType**;
    using reference = CellType*&;
    
    /**
     * @brief Construct a new Interior Cell Iterator
     * 
     * Creates an iterator for traversing interior cells of a mesh.
     * If the mesh has no interior cells (nx <= 2 or ny <= 2), 
     * this iterator will immediately be in the "end" state.
     * 
     * @param mesh Reference to the mesh to iterate over
     * @param i Initial i-index (default=1, first potential interior cell)
     * @param j Initial j-index (default=1, first potential interior cell)
     */
    InteriorCellIterator(MeshType& mesh, int i = 1, int j = 1)
        : m_mesh(mesh), m_i(i), m_j(j), m_current(nullptr) {
        
        // Early exit: Check if this mesh has any interior cells
        if (m_mesh.nx() <= 2 || m_mesh.ny() <= 2) {
            // No interior cells possible - set to end state
            setToEndState();
            return;
        }
        
        // For normal begin iterator (default arguments)
        if (i == 1 && j == 1) {
            // Try to find first valid interior cell
            m_current = m_mesh.getTypedCell(m_i, m_j);
            if (!m_current) {
                // First potential interior cell is null, find next valid one
                findNextValidCell();
            }
        } else {
            // For custom position or end iterator
            // Ensure the position is within interior bounds
            if (isInteriorPosition(i, j)) {
                m_current = m_mesh.getTypedCell(i, j);
            } else {
                // Position outside interior region - set to end state
                setToEndState();
            }
        }
    }
    
    /**
     * @brief Get the current cell
     * 
     * @return Pointer to the current cell (nullptr if at end)
     */
    value_type operator*() const { 
        return m_current; 
    }
    
    /**
     * @brief Advance to the next interior cell
     * 
     * If already at the end, does nothing.
     * 
     * @return Reference to this iterator after advancement
     */
    InteriorCellIterator& operator++() {
        // Safety check: If already at end, do nothing
        if (!m_current) {
            return *this;
        }
        
        // Find next valid interior cell
        findNextValidCell();
        return *this;
    }
    
    /**
     * @brief Post-increment operator
     * 
     * @return Copy of iterator before advancement
     */
    InteriorCellIterator operator++(int) {
        InteriorCellIterator tmp = *this;
        ++(*this);
        return tmp;
    }
    
    /**
     * @brief Equality comparison
     * 
     * Two iterators are equal if:
     * 1. Both are at the end state (m_current == nullptr), OR
     * 2. Both point to the same position in the mesh
     * 
     * @param other Iterator to compare with
     * @return true if iterators are equal
     */
    bool operator==(const InteriorCellIterator& other) const {
        // End state: either both are null or both are at the same "past-the-end" position
        if (!m_current && !other.m_current) {
            return true;
        }
        
        // Normal comparison - check indices match
        return m_i == other.m_i && m_j == other.m_j;
    }
    
    /**
     * @brief Inequality comparison
     * 
     * @param other Iterator to compare with
     * @return true if iterators are not equal
     */
    bool operator!=(const InteriorCellIterator& other) const {
        return !(*this == other);
    }
    
    /**
     * @brief Get current i-index
     * @return Current i-index
     */
    int i() const { return m_i; }
    
    /**
     * @brief Get current j-index
     * @return Current j-index
     */
    int j() const { return m_j; }
    
private:
    /**
     * @brief Set iterator to the end state
     * 
     * The end state is signified by:
     * 1. m_current = nullptr
     * 2. Position set to just past the last interior cell
     */
    void setToEndState() {
        m_current = nullptr;
        // Set position to just after last interior cell
        // (exact values don't matter since m_current is null,
        // but we use a consistent end position for comparison)
        m_i = 1;
        m_j = m_mesh.ny() - 1;
    }
    
    /**
     * @brief Check if a position is in the interior region
     * 
     * Interior positions satisfy: i > 0 && i < nx-1 && j > 0 && j < ny-1
     * 
     * @param i i-index to check
     * @param j j-index to check
     * @return true if position is in interior region
     */
    bool isInteriorPosition(int i, int j) const {
        return i > 0 && i < m_mesh.nx() - 1 && 
               j > 0 && j < m_mesh.ny() - 1;
    }
    
    /**
     * @brief Find the next valid interior cell
     * 
     * Advances the iterator's position until it finds a valid interior cell
     * or determines there are no more interior cells to visit (setting to end state).
     * 
     * The traversal follows a row-major order: incrementing i within each row,
     * then moving to the next row when the current row is exhausted.
     */
    void findNextValidCell() {
        // Safety check - ensure mesh can have interior cells
        if (m_mesh.nx() <= 2 || m_mesh.ny() <= 2) {
            setToEndState();
            return;
        }
        
        // Start from current position and find next valid cell
        do {
            ++m_i;
            // If reached the end of a row, move to next row
            if (m_i >= m_mesh.nx() - 1) {
                m_i = 1;  // First interior position in x
                ++m_j;
                
                // If we've gone past the last interior row, we're done
                if (m_j >= m_mesh.ny() - 1) {
                    setToEndState();
                    return;
                }
            }
            
            // Verify the new position is valid and has a cell
            if (isInteriorPosition(m_i, m_j)) {
                m_current = m_mesh.getTypedCell(m_i, m_j);
                if (m_current) {
                    return;  // Found valid cell
                }
            }
            
        } while (m_j < m_mesh.ny() - 1);  // Continue until we've checked all interior rows
        
        // If we get here, we've exhausted all possibilities
        setToEndState();
    }
    
    MeshType& m_mesh;      ///< Reference to the mesh being iterated
    int m_i;               ///< Current i-index
    int m_j;               ///< Current j-index
    CellType* m_current;   ///< Pointer to current cell (nullptr for end iterator)
};
  
  /**
   * @brief Iterator for boundary cells only
   * 
   * This iterator traverses only the cells at the perimeter of the mesh,
   * ensuring that each boundary cell is visited exactly once.
   */
  class BlockCellIterator {
  public:
    // STL iterator type traits
    using iterator_category = std::forward_iterator_tag;
    using value_type = CellType*;
    using difference_type = std::ptrdiff_t;
    using pointer = CellType**;
    using reference = CellType*&;
        
    /**
     * @brief Construct a new Block Cell Iterator
     * 
     * @param mesh Reference to the mesh
     * @param blockSizeX Block width
     * @param blockSizeY Block height
     * @param blockX Block x-index
     * @param blockY Block y-index
     * @param localI Local i-index within block
     * @param localJ Local j-index within block
     */
    BlockCellIterator(MeshType& mesh, 
		      int blockSizeX, int blockSizeY,
		      int blockX = 0, int blockY = 0,
		      int localI = 0, int localJ = 0)
      : m_mesh(mesh), 
	m_blockSizeX(blockSizeX), m_blockSizeY(blockSizeY),
	m_blockX(blockX), m_blockY(blockY),
	m_localI(localI), m_localJ(localJ),
	m_numBlocksX((mesh.nx() + blockSizeX - 1) / blockSizeX),
	m_numBlocksY((mesh.ny() + blockSizeY - 1) / blockSizeY) {
      // Find first valid cell if necessary
      if (blockX == 0 && blockY == 0 && localI == 0 && localJ == 0) {
	if (!m_mesh.getTypedCell(i(), j())) {
	  findNextValidCell();
	}
      }
    }
        
    // Core iterator operations
    value_type operator*() const { return m_mesh.getTypedCell(i(), j()); }
        
    BlockCellIterator& operator++() {
      ++m_localI;
            
      // If we reach the end of a block row or the mesh boundary
      if (m_localI >= m_blockSizeX || i() >= m_mesh.nx()) {
	m_localI = 0;
	++m_localJ;
                
	// If we reach the end of a block or the mesh boundary
	if (m_localJ >= m_blockSizeY || j() >= m_mesh.ny()) {
	  m_localJ = 0;
	  ++m_blockX;
                    
	  // If we reach the end of a row of blocks
	  if (m_blockX >= m_numBlocksX) {
	    m_blockX = 0;
	    ++m_blockY;
	  }
	}
      }
            
      // Find next valid cell if necessary
      if (m_blockY < m_numBlocksY && !m_mesh.getTypedCell(i(), j())) {
	findNextValidCell();
      }
            
      return *this;
    }
        
    BlockCellIterator operator++(int) {
      BlockCellIterator tmp = *this;
      ++(*this);
      return tmp;
    }
        
    // Comparison
    bool operator==(const BlockCellIterator& other) const {
      return m_blockX == other.m_blockX && 
	m_blockY == other.m_blockY &&
	m_localI == other.m_localI && 
	m_localJ == other.m_localJ;
    }
        
    bool operator!=(const BlockCellIterator& other) const {
      return !(*this == other);
    }
        
    // Current indices
    int i() const { return m_blockX * m_blockSizeX + m_localI; }
    int j() const { return m_blockY * m_blockSizeY + m_localJ; }
        
  private:
    void findNextValidCell() {
      while (m_blockY < m_numBlocksY) {
	if (i() < m_mesh.nx() && j() < m_mesh.ny() && m_mesh.getTypedCell(i(), j())) {
	  return;
	}
                
	++m_localI;
                
	// If we reach the end of a block row or the mesh boundary
	if (m_localI >= m_blockSizeX || i() >= m_mesh.nx()) {
	  m_localI = 0;
	  ++m_localJ;
                    
	  // If we reach the end of a block or the mesh boundary
	  if (m_localJ >= m_blockSizeY || j() >= m_mesh.ny()) {
	    m_localJ = 0;
	    ++m_blockX;
                        
	    // If we reach the end of a row of blocks
	    if (m_blockX >= m_numBlocksX) {
	      m_blockX = 0;
	      ++m_blockY;
	    }
	  }
	}
      }
    }
        
    MeshType& m_mesh;
    int m_blockSizeX, m_blockSizeY;
    int m_blockX, m_blockY;
    int m_localI, m_localJ;
    int m_numBlocksX, m_numBlocksY;
  };
    
  /**
   * @brief Range class for all cells
   */
  class CellRange {
  public:
    using iterator = CellIterator;
        
    CellRange(MeshType& mesh) : m_mesh(mesh) {}
        
    iterator begin() { return iterator(m_mesh); }
    iterator end() { return iterator(m_mesh, 0, m_mesh.ny()); }
        
  private:
    MeshType& m_mesh;
  };
    
  /**
   * @brief Range class for interior cells
   */
  class InteriorCellRange {
  public:
    using iterator = InteriorCellIterator;
        
    InteriorCellRange(MeshType& mesh) : m_mesh(mesh) {}
        
    iterator begin() { return iterator(m_mesh); }
    iterator end() { return iterator(m_mesh, 1, m_mesh.ny() - 1); }
        
  private:
    MeshType& m_mesh;
  };
    
  /**
   * @brief Range class for boundary cells
   */
  class BoundaryCellRange {
  public:
    using iterator = BoundaryCellIterator;
        
    BoundaryCellRange(MeshType& mesh) : m_mesh(mesh) {}
        
    iterator begin() { return iterator(m_mesh); }
    iterator end() { return iterator(m_mesh, 4, 0); }
        
  private:
    MeshType& m_mesh;
  };
    
  /**
   * @brief Range class for blocked cell traversal
   */
  class BlockCellRange {
  public:
    using iterator = BlockCellIterator;
        
    BlockCellRange(MeshType& mesh, int blockSizeX = 16, int blockSizeY = 16) 
      : m_mesh(mesh), m_blockSizeX(blockSizeX), m_blockSizeY(blockSizeY) {}
        
    iterator begin() { 
      return iterator(m_mesh, m_blockSizeX, m_blockSizeY); 
    }
        
    iterator end() { 
      int numBlocksY = (m_mesh.ny() + m_blockSizeY - 1) / m_blockSizeY;
      return iterator(m_mesh, m_blockSizeX, m_blockSizeY, 0, numBlocksY); 
    }
        
  private:
    MeshType& m_mesh;
    int m_blockSizeX, m_blockSizeY;
  };
};

#endif // MESH_ACCESSOR_ITERATORS_H
