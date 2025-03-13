
#include "Mesh.h"

namespace mesh {
  int Cell::linearIndex() const {
    if (!isValid()) {
      throw std::logic_error("Cannot compute linear index: invalid cell");
    }
    return m_mesh->linearIndex(m_i, m_j);
  }

  bool Cell::isBoundary() const {
    if (!isValid()) {
      throw std::logic_error("Cannot determine boundary status: invalid cell");
    }
    return (m_i == 0 || m_j == 0 || m_i == m_mesh->nx() - 1 || m_j == m_mesh->ny() - 1);
  }

  Cell Cell::neighbor(uint8_t direction) const {
    if (!isValid()) {
      throw std::logic_error("Cannot get neighbor: invalid cell");
    }

    auto [di, dj] = getDirectionOffset(direction);
    int ni = m_i + di;
    int nj = m_j + dj;

    if (ni < 0 || ni >= m_mesh->nx() || nj < 0 || nj >= m_mesh->ny()) {
      return Cell(nullptr, -1, -1);
    }

    return Cell(m_mesh, ni, nj);
  }

  bool Cell::isValid() const {
    return m_mesh != nullptr && m_i >= 0 && m_j >= 0 &&
           m_i < m_mesh->nx() && m_j < m_mesh->ny();
  }
}

