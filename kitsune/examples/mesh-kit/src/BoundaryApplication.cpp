#include "BoundaryZone.h"
#include "Mesh.h"

void Mesh::applyAllBoundaryConditions(double time, double dt) {
  // Apply boundary conditions for all zones
  for (const auto &[name, zone] : m_boundaryZones) {
    if (zone->isActive()) {
      zone->applyBoundaryCondition(*this, time, dt);
    }
  }
}

// This function can be used for sequential application of boundary conditions
// when the order matters (e.g., for complex interdependent boundaries)
void applyBoundaryConditionsInOrder(Mesh &mesh,
                                    const std::vector<std::string> &zoneOrder,
                                    double time, double dt) {

  // Apply boundary conditions in specified order
  for (const auto &zoneName : zoneOrder) {
    auto zone = mesh.getBoundaryZone(zoneName);
    if (zone && zone->isActive()) {
      zone->applyBoundaryCondition(mesh, time, dt);
    }
  }
}

// Apply a specific type of boundary condition across all zones
template <typename BoundaryType>
void applyBoundaryTypeToMesh(Mesh &mesh, double time, double dt) {
  for (const auto &zone : mesh.getBoundaryZones()) {
    auto bc = zone->getBoundaryCondition();

    // Check if the boundary condition is of the specified type
    if (dynamic_cast<BoundaryType *>(bc.get()) && zone->isActive()) {
      zone->applyBoundaryCondition(mesh, time, dt);
    }
  }
}