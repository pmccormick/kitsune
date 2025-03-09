/**
 * @file CellBase.cpp
 * @brief Implementation of the CellBase class with bit field directions
 * 
 * Design Agenda:
 * --------------
 * This implementation uses a bit field approach for several key reasons:
 * 
 * 1. Performance: Using bit operations and constexpr enables more efficient
 *    directional operations with compile-time optimizations
 * 
 * 2. Memory efficiency: Compact representation of directions reduces memory
 *    overhead in large-scale simulations
 * 
 * 3. Extensibility: The bit field approach makes it easier to add diagonal 
 *    directions and potentially extend to 3D in the future
 * 
 * 4. Simplicity: We deliberately avoid complex template metaprogramming to
 *    keep compilation times reasonable while still providing significant
 *    optimizations
 * 
 * Field Access Strategy:
 * ----------------------
 * The implementation is designed to work with generated subclasses that
 * provide direct field access. These specialized subclasses avoid the
 * overhead of virtual function calls and type erasure by directly accessing
 * the fields stored in the mesh.
 */

#include "CellBase.h"
#include "MeshBase.h"

CellBase::CellBase(int i, int j, MeshBase* mesh)
    : m_i(i), m_j(j), m_mesh(mesh) 
{
}

int CellBase::i() const 
{
    return m_i;
}

int CellBase::j() const 
{
    return m_j;
}

MeshBase* CellBase::mesh() const 
{
    return m_mesh;
}

int CellBase::linearIndex() const 
{
    return m_mesh->linearIndex(m_i, m_j);
}

std::pair<double, double> CellBase::position() const 
{
    return m_mesh->position(m_i, m_j);
}

bool CellBase::isBoundary() const 
{
    return m_i == 0 || m_j == 0 || m_i == m_mesh->nx() - 1 || m_j == m_mesh->ny() - 1;
}

CellBase* CellBase::neighbor(uint8_t direction) const 
{
    auto [ni, nj] = neighborIndices(direction);
    
    // Check if the neighbor is within mesh bounds
    if (ni >= 0 && ni < m_mesh->nx() && nj >= 0 && nj < m_mesh->ny()) {
        return m_mesh->getCell(ni, nj);
    }
    
    return nullptr;
}

CellBase* CellBase::neighborByIndex(int directionIndex) const
{
    // Map the direction index to the corresponding bit flag
    uint8_t directionFlag;
    switch (directionIndex) {
        case NORTH_IDX: directionFlag = NORTH; break;
        case EAST_IDX:  directionFlag = EAST;  break;
        case SOUTH_IDX: directionFlag = SOUTH; break;
        case WEST_IDX:  directionFlag = WEST;  break;
        default:        directionFlag = NONE;  break;
    }
    
    return neighbor(directionFlag);
}

std::array<CellBase*, 4> CellBase::neighbors() const 
{
    std::array<CellBase*, 4> result;
    
    // Optimized implementation using the pre-computed direction offsets
    for (int dir = 0; dir < 4; ++dir) {
        const auto [di, dj] = PRIMARY_DIRECTION_OFFSETS[dir];
        int ni = m_i + di;
        int nj = m_j + dj;
        
        // Check if the neighbor is within mesh bounds
        if (ni >= 0 && ni < m_mesh->nx() && nj >= 0 && nj < m_mesh->ny()) {
            result[dir] = m_mesh->getCell(ni, nj);
        } else {
            result[dir] = nullptr;
        }
    }
    
    return result;
}

std::vector<CellBase*> CellBase::getNeighbors(uint8_t directions) const
{
    std::vector<CellBase*> result;
    
    // Check each primary direction
    if (directions & NORTH) {
        CellBase* northNeighbor = neighbor(NORTH);
        if (northNeighbor) result.push_back(northNeighbor);
    }
    
    if (directions & EAST) {
        CellBase* eastNeighbor = neighbor(EAST);
        if (eastNeighbor) result.push_back(eastNeighbor);
    }
    
    if (directions & SOUTH) {
        CellBase* southNeighbor = neighbor(SOUTH);
        if (southNeighbor) result.push_back(southNeighbor);
    }
    
    if (directions & WEST) {
        CellBase* westNeighbor = neighbor(WEST);
        if (westNeighbor) result.push_back(westNeighbor);
    }
    
    return result;
}

std::pair<int, int> CellBase::neighborIndices(uint8_t direction) const 
{
    // Use constexpr function for compile-time optimization when direction is a constant
    int ni = m_i;
    int nj = m_j;
    
    // For primary directions, we can optimize using a direct array lookup
    if (direction == NORTH) {
        return {m_i, m_j + 1};
    } else if (direction == EAST) {
        return {m_i + 1, m_j};
    } else if (direction == SOUTH) {
        return {m_i, m_j - 1};
    } else if (direction == WEST) {
        return {m_i - 1, m_j};
    }
    
    // For combined directions, we apply the offsets
    const auto [di, dj] = getDirectionOffset(direction);
    return {m_i + di, m_j + dj};
}

std::pair<int, int> CellBase::locationIndices(int location, uint8_t direction) const 
{
    int li = m_i;
    int lj = m_j;
    
    switch (location) {
        case CELL_CENTER:
            // No adjustment needed for cell center
            break;
            
        case CELL_VERTEX:
            // Adjust for vertices using bitwise checks for optimization
            if (direction & EAST) li += 1;  // East-side vertices
            if (direction & NORTH) lj += 1; // North-side vertices
            break;
            
        case HORIZONTAL_EDGE:
            // For horizontal edges, adjust j index
            if (direction & NORTH) lj += 1;  // North edge
            break;
            
        case VERTICAL_EDGE:
            // For vertical edges, adjust i index
            if (direction & EAST) li += 1;  // East edge
            break;
    }
    
    return {li, lj};
}

