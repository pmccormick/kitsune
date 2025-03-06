#pragma once

#include "Cell.h"
#include "Mesh.h"
#include <array>
#include <vector>

/**
 * @class MeshTopologyAdapter
 * @brief Provides topology and geometric information without modifying existing classes
 * 
 * This adapter class provides an interface to access mesh topology and geometric 
 * information needed for boundary condition assignment and mesh validation,
 * without modifying the performance-optimized Cell and Mesh classes.
 * 
 * Design considerations:
 * 1. No state modification of underlying objects
 * 2. Computation over storage where appropriate for GPU compatibility
 * 3. Optional caching for repeated operations
 * 4. Support for parallel operations
 * 5. Direct indexing methods to avoid indirection
 */
class MeshTopologyAdapter {
public:
    /**
     * @brief Constructor
     * @param mesh Reference to the mesh
     */
    MeshTopologyAdapter(Mesh& mesh);
    
    /**
     * @struct CellGeometry
     * @brief Contains cached geometric information for a cell
     */
    struct CellGeometry {
        double area;
        std::array<double, 2> centroid;
        std::array<std::array<double, 2>, 4> vertices;
    };
    
    /**
     * @struct FaceGeometry
     * @brief Contains cached geometric information for a face
     */
    struct FaceGeometry {
        std::array<double, 2> normal;
        std::array<double, 2> centroid;
        std::array<std::array<double, 2>, 2> vertices;
        double length;
    };
    
    /**
     * @brief Get cell centroid in physical coordinates
     * @param i Cell i-index
     * @param j Cell j-index
     * @return Array containing [x, y] coordinates
     */
    std::array<double, 2> getCellCentroid(size_t i, size_t j) const;
    
    /**
     * @brief Get cell vertices in physical coordinates
     * @param i Cell i-index
     * @param j Cell j-index
     * @return Array of vertex coordinates [x, y]
     */
    std::array<std::array<double, 2>, 4> getCellVertices(size_t i, size_t j) const;
    
    /**
     * @brief Get cell area
     * @param i Cell i-index
     * @param j Cell j-index
     * @return Cell area
     */
    double getCellArea(size_t i, size_t j) const;
    
    /**
     * @brief Get neighboring cell indices
     * @param i Cell i-index
     * @param j Cell j-index
     * @return Array of neighbor indices [i, j] (may contain invalid indices if on boundary)
     */
    std::array<std::array<size_t, 2>, 4> getNeighborIndices(size_t i, size_t j) const;
    
    /**
     * @brief Get valid neighboring cell indices
     * @param i Cell i-index
     * @param j Cell j-index
     * @return Vector of valid neighbor indices [i, j]
     */
    std::vector<std::array<size_t, 2>> getValidNeighborIndices(size_t i, size_t j) const;
    
    /**
     * @brief Check if a cell is on the domain boundary
     * @param i Cell i-index
     * @param j Cell j-index
     * @return True if cell is on any boundary
     */
    bool isBoundaryCell(size_t i, size_t j) const;
    
    /**
     * @brief Get the face normal between two cells
     * @param i1 First cell i-index
     * @param j1 First cell j-index
     * @param i2 Second cell i-index
     * @param j2 Second cell j-index
     * @return Normal vector [nx, ny] pointing from cell 1 to cell 2
     */
    std::array<double, 2> getFaceNormal(size_t i1, size_t j1, size_t i2, size_t j2) const;
    
    /**
     * @brief Get face centroid between two cells
     * @param i1 First cell i-index
     * @param j1 First cell j-index
     * @param i2 Second cell i-index
     * @param j2 Second cell j-index
     * @return Face centroid [x, y]
     */
    std::array<double, 2> getFaceCentroid(size_t i1, size_t j1, size_t i2, size_t j2) const;
    
    /**
     * @brief Get face vertices between two cells
     * @param i1 First cell i-index
     * @param j1 First cell j-index
     * @param i2 Second cell i-index
     * @param j2 Second cell j-index
     * @return Array of two vertex coordinates [x, y]
     */
    std::array<std::array<double, 2>, 2> getFaceVertices(size_t i1, size_t j1, size_t i2, size_t j2) const;
    
    /**
     * @brief Get face length between two cells
     * @param i1 First cell i-index
     * @param j1 First cell j-index
     * @param i2 Second cell i-index
     * @param j2 Second cell j-index
     * @return Face length
     */
    double getFaceLength(size_t i1, size_t j1, size_t i2, size_t j2) const;
    
    /**
     * @brief Get all boundary cell indices
     * @return Vector of boundary cell indices [i, j]
     */
    std::vector<std::array<size_t, 2>> getAllBoundaryCells() const;
    
    /**
     * @brief Get boundary cells on a specific edge
     * @param edge Edge identifier (0=bottom, 1=right, 2=top, 3=left)
     * @return Vector of boundary cell indices [i, j]
     */
    std::vector<std::array<size_t, 2>> getBoundaryEdgeCells(int edge) const;
    
    /**
     * @brief Get external face normal for a boundary cell
     * @param i Cell i-index
     * @param j Cell j-index
     * @return Normal vector [nx, ny] pointing outward from the domain
     */
    std::array<double, 2> getBoundaryFaceNormal(size_t i, size_t j) const;
    
    /**
     * @brief Calculate non-orthogonality between neighboring cells
     * @param i1 First cell i-index
     * @param j1 First cell j-index
     * @param i2 Second cell i-index
     * @param j2 Second cell j-index
     * @return Angle in radians
     */
    double calculateNonOrthogonality(size_t i1, size_t j1, size_t i2, size_t j2) const;
    
    /**
     * @brief Calculate aspect ratio of a cell
     * @param i Cell i-index
     * @param j Cell j-index
     * @return Aspect ratio (ratio of longest to shortest dimension)
     */
    double calculateAspectRatio(size_t i, size_t j) const;
    
    /**
     * @brief Calculate skewness of a face
     * @param i1 First cell i-index
     * @param j1 First cell j-index
     * @param i2 Second cell i-index
     * @param j2 Second cell j-index
     * @return Skewness measure (0-1 scale)
     */
    double calculateSkewness(size_t i1, size_t j1, size_t i2, size_t j2) const;
    
    /**
     * @brief Calculate volume ratio between neighboring cells
     * @param i1 First cell i-index
     * @param j1 First cell j-index
     * @param i2 Second cell i-index
     * @param j2 Second cell j-index
     * @return Ratio (always >= 1.0)
     */
    double calculateVolumeRatio(size_t i1, size_t j1, size_t i2, size_t j2) const;
    
    /**
     * @brief Batch compute geometry for all cells (for parallel execution)
     * @param useCache Whether to store results in cache
     */
    void computeAllCellGeometry(bool useCache = true);
    
    /**
     * @brief Batch compute face geometry for all internal faces (for parallel execution)
     * @param useCache Whether to store results in cache
     */
    void computeAllFaceGeometry(bool useCache = true);
    
    /**
     * @brief Clear cached geometry data
     */
    void clearCache();
    
private:
    Mesh& m_mesh;
    
    // Optional caching structures
    bool m_useCache;
    std::vector<CellGeometry> m_cellGeometryCache;
    std::unordered_map<uint64_t, FaceGeometry> m_faceGeometryCache;
    
    // Helper to create face key from cell indices
    uint64_t makeFaceKey(size_t i1, size_t j1, size_t i2, size_t j2) const;
    
    // Check if cell indices are valid
    bool isValidCell(size_t i, size_t j) const;
    
    // Check if cells are neighbors
    bool areCellsNeighbors(size_t i1, size_t j1, size_t i2, size_t j2) const;
};

