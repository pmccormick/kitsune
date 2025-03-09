/**
 * @file MeshAccessor.h
 * @brief Simple accessor for whole-mesh operations
 * 
 * This header defines a basic accessor for performing operations on an entire mesh
 * without region filtering or complex domain decomposition.
 */

#ifndef MESH_ACCESSOR_H
#define MESH_ACCESSOR_H

#include "Mesh.h"
#include <functional>

/**
 * @brief Simple accessor for whole-mesh operations
 * 
 * Provides straightforward access methods for operations on the entire mesh
 * without region filtering or complex domain decomposition.
 * 
 * @tparam MeshType The specialized Mesh type
 * @tparam CellType The cell type used by the mesh
 */
template <typename MeshType, typename CellType>
class MeshAccessor {
public:
    /**
     * @brief Construct a new Mesh Accessor
     * 
     * @param mesh Reference to the mesh
     */
    explicit MeshAccessor(MeshType& mesh) : m_mesh(mesh) {}
    
    /**
     * @brief Apply a function to all cells in the mesh
     * 
     * @param func Function to apply to each cell
     */
    void forEachCell(std::function<void(CellType*)> func) {
        for (int j = 0; j < m_mesh.ny(); ++j) {
            for (int i = 0; i < m_mesh.nx(); ++i) {
                CellType* cell = m_mesh.getCell(i, j);
                if (cell) {
                    func(cell);
                }
            }
        }
    }
    
    /**
     * @brief Apply a function to interior cells only
     * 
     * @param func Function to apply to each interior cell
     */
    void forEachInteriorCell(std::function<void(CellType*)> func) {
        for (int j = 1; j < m_mesh.ny() - 1; ++j) {
            for (int i = 1; i < m_mesh.nx() - 1; ++i) {
                CellType* cell = m_mesh.getCell(i, j);
                if (cell) {
                    func(cell);
                }
            }
        }
    }
    
    /**
     * @brief Apply a function to cells in cache-friendly blocks
     * 
     * @param func Function to apply to each cell
     * @param blockSize Size of blocks for cache optimization
     */
    void forEachCellBlocked(std::function<void(CellType*)> func, int blockSize = 16) {
        for (int jBlock = 0; jBlock < m_mesh.ny(); jBlock += blockSize) {
            for (int iBlock = 0; iBlock < m_mesh.nx(); iBlock += blockSize) {
                // Calculate the end of this block (clamping to mesh boundaries)
                int jEnd = std::min(jBlock + blockSize, m_mesh.ny());
                int iEnd = std::min(iBlock + blockSize, m_mesh.nx());
                
                // Process cells in this block
                for (int j = jBlock; j < jEnd; ++j) {
                    for (int i = iBlock; i < iEnd; ++i) {
                        CellType* cell = m_mesh.getCell(i, j);
                        if (cell) {
                            func(cell);
                        }
                    }
                }
            }
        }
    }
    
    /**
     * @brief Apply a function to boundary cells only
     * 
     * @param func Function to apply to each boundary cell
     */
    void forEachBoundaryCell(std::function<void(CellType*)> func) {
        const int nx = m_mesh.nx();
        const int ny = m_mesh.ny();
        
        // Top and bottom rows
        for (int i = 0; i < nx; ++i) {
            CellType* bottomCell = m_mesh.getCell(i, 0);
            if (bottomCell) {
                func(bottomCell);
            }
            
            CellType* topCell = m_mesh.getCell(i, ny - 1);
            if (topCell) {
                func(topCell);
            }
        }
        
        // Left and right columns (excluding corners which are already processed)
        for (int j = 1; j < ny - 1; ++j) {
            CellType* leftCell = m_mesh.getCell(0, j);
            if (leftCell) {
                func(leftCell);
            }
            
            CellType* rightCell = m_mesh.getCell(nx - 1, j);
            if (rightCell) {
                func(rightCell);
            }
        }
    }
    
    /**
     * @brief Apply a stencil operation to each interior cell
     * 
     * A stencil operation takes a center cell and its neighbors
     * and produces a result.
     * 
     * @tparam StencilOp Type of stencil operation
     * @param stencilOp Stencil operation to apply
     */
    template <typename StencilOp>
    void applyStencil(StencilOp stencilOp) {
        for (int j = 1; j < m_mesh.ny() - 1; ++j) {
            for (int i = 1; i < m_mesh.nx() - 1; ++i) {
                CellType* center = m_mesh.getCell(i, j);
                CellType* north = m_mesh.getCell(i, j+1);
                CellType* east = m_mesh.getCell(i+1, j);
                CellType* south = m_mesh.getCell(i, j-1);
                CellType* west = m_mesh.getCell(i-1, j);
                
                if (center && north && east && south && west) {
                    stencilOp(center, north, east, south, west);
                }
            }
        }
    }
    
    /**
     * @brief Get the mesh being accessed
     * 
     * @return MeshType& Reference to the mesh
     */
    MeshType& mesh() { return m_mesh; }
    
    /**
     * @brief Get const reference to the mesh
     * 
     * @return const MeshType& Const reference to the mesh
     */
    const MeshType& mesh() const { return m_mesh; }

private:
    MeshType& m_mesh;  ///< Reference to the mesh
};

#endif // MESH_ACCESSOR_H


