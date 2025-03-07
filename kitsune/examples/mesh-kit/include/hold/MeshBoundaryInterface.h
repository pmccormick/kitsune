#pragma once

#include "BoundaryZone.h"
#include "Mesh.h"
#include "Vector2D.h"
#include <memory>
#include <string>
#include <unordered_map>

/**
 * @brief Enhancement to the Mesh class to support boundary management
 *
 * This assumes the Mesh class already exists and provides basic mesh
 * functionality. The following methods should be added to the Mesh class.
 */
class Mesh {
public:
  // Existing Mesh methods...

  /**
   * @brief Get a reference to a face by its ID
   * @param faceID ID of the face
   * @return Reference to the face
   * @throws std::out_of_range if the face doesn't exist
   */
  Face &getFace(MeshFaceID faceID);

  /**
   * @brief Get a reference to a boundary cell associated with a face
   * @param faceID ID of the boundary face
   * @return Reference to the boundary cell
   * @throws std::out_of_range if the face isn't a boundary face
   */
  Cell &getBoundaryCell(MeshFaceID faceID);

  /**
   * @brief Get neighboring interior cells for a boundary face
   * @param faceID ID of the boundary face
   * @return Vector of pointers to neighboring cells
   */
  std::vector<Cell *> getNeighboringCells(MeshFaceID faceID);

  /**
   * @brief Get all boundary faces in the mesh
   * @return Vector of IDs for all boundary faces
   */
  std::vector<MeshFaceID> getBoundaryFaces() const;

  /**
   * @brief Get boundary faces by named region
   * @param regionName Name of the boundary region
   * @return Vector of face IDs in the named region
   */
  std::vector<MeshFaceID>
  getBoundaryFacesByRegion(const std::string &regionName) const;

  /**
   * @brief Add a boundary zone to the mesh
   * @param zone Shared pointer to the boundary zone
   */
  void addBoundaryZone(std::shared_ptr<BoundaryZone> zone);

  /**
   * @brief Get a boundary zone by name
   * @param name Name of the boundary zone
   * @return Shared pointer to the boundary zone, or nullptr if not found
   */
  std::shared_ptr<BoundaryZone> getBoundaryZone(const std::string &name) const;

  /**
   * @brief Get all boundary zones
   * @return Vector of shared pointers to all boundary zones
   */
  std::vector<std::shared_ptr<BoundaryZone>> getBoundaryZones() const;

  /**
   * @brief Get the normal vector of a face
   * @param faceID ID of the face
   * @return Normal vector of the face (outward facing for boundary faces)
   */
  Vector2D getFaceNormal(MeshFaceID faceID) const;

  /**
   * @brief Apply all boundary conditions
   * @param time Current simulation time
   * @param dt Time step size
   */
  void applyAllBoundaryConditions(double time, double dt);

private:
  // Add these members to the Mesh class:

  // Map of boundary zones by name
  std::unordered_map<std::string, std::shared_ptr<BoundaryZone>>
      m_boundaryZones;

  // Map of boundary faces to their regions
  std::unordered_map<std::string, std::vector<MeshFaceID>> m_boundaryRegions;
};