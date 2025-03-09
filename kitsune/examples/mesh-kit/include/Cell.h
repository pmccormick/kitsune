/**
 * @file CellBase.h
 * @brief Lightweight base class for cell operations with optimized directional components
 * 
 * Design Agenda:
 * --------------
 * This implementation uses a bit field approach for representing
 * directions, offering several key benefits:
 * 
 * 1. Memory efficiency through compact representation of directions
 * 2. Improved performance via compile-time optimizations with constexpr
 * 3. Enhanced flexibility by supporting combined directions (diagonals)
 * 4. Future extensibility for 3D directions if needed
 * 
 * The design intentionally avoids complex template metaprogramming to keep
 * compilation times reasonable while still providing significant performance
 * optimizations. We leverage constexpr to enable the compiler to perform
 * calculations at compile time when directions are known constants.
 * 
 * Field Access Strategy:
 * ----------------------
 * This base class is designed to work with a code generation system that creates
 * specialized subclasses with direct field access capabilities. These generated
 * subclasses provide strongly-typed, direct access to field data without the
 * overhead of virtual function calls or string-based field lookups.
 * 
 * For example, a generated cell might implement:
 *   double temperature() const;       // Direct access to temperature field
 *   void setTemperature(double val);  // Direct update to temperature field
 * 
 * This approach maintains a lightweight interface for access to cell indices,
 * neighbors, and related properties, while improving performance and reducing
 * memory overhead in large-scale simulations.
 */

#ifndef CELL_BASE_H
#define CELL_BASE_H

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

// Forward declarations
class MeshBase;

/**
 * @class CellBase
 * @brief Lightweight base class for cell operations with bit field directions
 * 
 * CellBase provides a minimal interface for accessing cell indices,
 * neighbors, and related properties. It acts as an iterator-like concept,
 * allowing code to "walk" through mesh elements without directly dealing
 * with the complexity of index manipulation and data structure details.
 * 
 * This implementation uses bit fields for direction representation to
 * enable more efficient operations and combined directions.
 * 
 * @note Specialized subclasses will be generated with direct field access methods
 *       based on mesh configuration. These provide the most efficient means of
 *       accessing field data and should be used whenever possible.
 */
class CellBase {
public:
    /**
     * @brief Direction bit flags for neighbor access
     * 
     * Each direction is represented as a bit in a uint8_t value,
     * allowing for combined directions (e.g., NORTHEAST = NORTH | EAST)
     */
    static constexpr uint8_t NONE      = 0x00; ///< No direction (0000 0000)
    static constexpr uint8_t NORTH     = 0x01; ///< North direction (0000 0001)
    static constexpr uint8_t EAST      = 0x02; ///< East direction (0000 0010)
    static constexpr uint8_t SOUTH     = 0x04; ///< South direction (0000 0100)
    static constexpr uint8_t WEST      = 0x08; ///< West direction (0000 1000)
    
    // Combined directions for convenience
    static constexpr uint8_t NORTHEAST = NORTH | EAST; ///< Northeast diagonal (0000 0011)
    static constexpr uint8_t SOUTHEAST = SOUTH | EAST; ///< Southeast diagonal (0000 0110)
    static constexpr uint8_t SOUTHWEST = SOUTH | WEST; ///< Southwest diagonal (0000 1100)
    static constexpr uint8_t NORTHWEST = NORTH | WEST; ///< Northwest diagonal (0000 1001)
    
    // Aliases for traditional direction indices (for backward compatibility)
    static constexpr int NORTH_IDX = 0;
    static constexpr int EAST_IDX  = 1;
    static constexpr int SOUTH_IDX = 2;
    static constexpr int WEST_IDX  = 3;
    
    /**
     * @brief Location types for field data
     */
    static constexpr int CELL_CENTER = 0;
    static constexpr int CELL_VERTEX = 1;
    static constexpr int HORIZONTAL_EDGE = 2;
    static constexpr int VERTICAL_EDGE = 3;
    
    /**
     * @brief Get direction offset for a given direction
     * 
     * @param direction Direction bit flag
     * @return std::pair<int, int> (di, dj) offset for the direction
     */
    static constexpr std::pair<int, int> getDirectionOffset(uint8_t direction) {
        int di = 0;
        int dj = 0;
        
        if (direction & NORTH) dj += 1;
        if (direction & SOUTH) dj -= 1;
        if (direction & EAST)  di += 1;
        if (direction & WEST)  di -= 1;
        
        return {di, dj};
    }
    
    /**
     * @brief Get offsets for primary directions
     * 
     * Pre-computed array of direction offsets for fast access to
     * the four primary directions without branches.
     */
    static constexpr std::array<std::pair<int, int>, 4> PRIMARY_DIRECTION_OFFSETS = {{
        {0, 1},   // NORTH
        {1, 0},   // EAST
        {0, -1},  // SOUTH
        {-1, 0}   // WEST
    }};
    
    /**
     * @brief Construct a new Cell Base object
     * 
     * @param i Index in x-direction
     * @param j Index in y-direction
     * @param mesh Pointer to the owning mesh
     */
    CellBase(int i, int j, MeshBase* mesh);
    
    /**
     * @brief Virtual destructor
     */
    virtual ~CellBase() = default;
    
    /**
     * @brief Get index in x-direction
     * 
     * @return int Index in x-direction
     */
    int i() const;
    
    /**
     * @brief Get index in y-direction
     * 
     * @return int Index in y-direction
     */
    int j() const;
    
    /**
     * @brief Get pointer to the owning mesh
     * 
     * @return MeshBase* Pointer to mesh
     */
    MeshBase* mesh() const;
    
    /**
     * @brief Convert to linear index for direct array access
     * 
     * @return int Linear index
     */
    int linearIndex() const;
    
    /**
     * @brief Get physical position of cell center
     * 
     * @return std::pair<double, double> (x,y) position
     */
    std::pair<double, double> position() const;
    
    /**
     * @brief Check if cell is at mesh boundary
     * 
     * @return true If the cell is at any boundary
     * @return false If the cell is interior
     */
    bool isBoundary() const;
    
    /**
     * @brief Get neighboring cell in specified direction
     * 
     * @param direction Direction bit flag (NORTH, EAST, etc.)
     * @return CellBase* Pointer to neighbor (nullptr if boundary)
     * 
     * @note Specialized subclasses may override this to return their specific type
     */
    virtual CellBase* neighbor(uint8_t direction) const;
    
    /**
     * @brief Get neighboring cell using traditional direction index
     * 
     * Provided for backward compatibility with existing code
     * 
     * @param directionIndex One of NORTH_IDX, EAST_IDX, SOUTH_IDX, WEST_IDX
     * @return CellBase* Pointer to neighbor (nullptr if boundary)
     */
    CellBase* neighborByIndex(int directionIndex) const;
    
    /**
     * @brief Get all neighbors in primary directions
     * 
     * @return std::array<CellBase*, 4> Array of neighbors (may include nullptr)
     */
    std::array<CellBase*, 4> neighbors() const;
    
    /**
     * @brief Get neighbors for specified directions
     * 
     * @param directions Direction bit flags (can be combined)
     * @return std::vector<CellBase*> Vector of neighbors (may include nullptr)
     */
    std::vector<CellBase*> getNeighbors(uint8_t directions) const;
    
    /**
     * @brief Get indices of neighbor in specified direction
     * 
     * @param direction Direction bit flag (NORTH, EAST, etc.)
     * @return std::pair<int, int> (i,j) indices of neighbor
     */
    std::pair<int, int> neighborIndices(uint8_t direction) const;
    
    /**
     * @brief Get indices for accessing data at a specific location
     * 
     * @param location One of CELL_CENTER, CELL_VERTEX, etc.
     * @param direction Direction bit flag for edges/vertices
     * @return std::pair<int, int> Adjusted indices for field access
     */
    std::pair<int, int> locationIndices(int location, uint8_t direction = NONE) const;

protected:
    int m_i;            ///< Index in x-direction
    int m_j;            ///< Index in y-direction
    MeshBase* m_mesh;   ///< Pointer to owning mesh
    
    /**
     * @brief Helper for generated field access methods
     * 
     * This method exposes the raw cell indices for use by specialized
     * field access methods in generated subclasses. It allows the 
     * code generator to create direct field access without needing to
     * go through virtual methods.
     * 
     * @param location Field location (CELL_CENTER, VERTEX, etc.)
     * @param direction Direction for non-centered fields (default = NONE)
     * @return Indices adjusted for the field location
     * 
     * @note This method is intended for use by generated code, not direct use.
     */
    std::pair<int, int> getFieldIndices(int location, uint8_t direction = NONE) const {
        return locationIndices(location, direction);
    }
};

#endif // CELL_BASE_H
 
