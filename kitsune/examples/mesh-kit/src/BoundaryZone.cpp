#include "BoundaryZone.h"
#include "Cell.h"
#include "Face.h"
#include "Mesh.h"
#include "Vector2D.h"

void BoundaryZone::applyBoundaryCondition(Mesh &mesh, double time, double dt) {
  if (!m_isActive || !m_boundaryCondition) {
    return;
  }

  for (const auto &faceID : m_boundaryFaces) {
    // Get the face and associated boundary cell
    Face &face = mesh.getFace(faceID);
    Cell &boundaryCell = mesh.getBoundaryCell(faceID);

    // Get the centroid coordinates of the face
    const Vector2D &centroid = face.getCentroid();

    // Get neighboring interior cells (if any)
    std::vector<Cell *> neighbors = mesh.getNeighboringCells(faceID);

    // Apply the boundary condition to the cell
    m_boundaryCondition->apply(boundaryCell, centroid.x, centroid.y, time,
                               &neighbors);
  }
}