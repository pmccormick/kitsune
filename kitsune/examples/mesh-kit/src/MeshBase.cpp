/**
 * @file MeshBase.cpp
 * @brief Implementation of the MeshBase abstract class
 */

#include "MeshBase.h"
#include <stdexcept>

MeshBase::MeshBase(int nx, int ny, double dx, double dy)
    : m_nx(nx), m_ny(ny), m_dx(dx), m_dy(dy)
{
    // Ensure valid dimensions
    if (nx <= 0 || ny <= 0) {
        throw std::invalid_argument("Mesh dimensions must be positive");
    }
    
    // Ensure valid grid spacing
    if (dx <= 0.0 || dy <= 0.0) {
        throw std::invalid_argument("Grid spacing must be positive");
    }
}


