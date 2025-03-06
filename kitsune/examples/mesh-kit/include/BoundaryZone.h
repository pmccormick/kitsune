#pragma once

#include "BoundaryClass.h"
#include <memory>
#include <string>
#include <vector>

// Forward declarations
class Mesh;
class Face;
class Cell;

/**
 * @brief Type alias for a mesh face identifier
 */
using MeshFaceID = int;

/**
 * @class BoundaryZone
 * @brief Links mesh boundary elements to a boundary condition
 *
 * A BoundaryZone establishes the connection between physical mesh boundaries
 * and the boundary conditions that should be applied to them. It stores
 * references to boundary faces but doesn't own them.
 */
class BoundaryZone {
private:
  std::string m_name;
  std::shared_ptr<BoundaryClass> m_boundaryCondition;
  std::vector<MeshFaceID> m_boundaryFaces; // Store IDs, not objects
  bool m_isActive;

public:
  /**
   * @brief Constructor
   * @param name Name of the boundary zone
   * @param bc Shared pointer to the boundary condition
   */
  BoundaryZone(const std::string &name, std::shared_ptr<BoundaryClass> bc)
      : m_name(name), m_boundaryCondition(bc), m_isActive(true) {}

  /**
   * @brief Get the name of the boundary zone
   * @return Name of the boundary zone
   */
  std::string getName() const { return m_name; }

  /**
   * @brief Add a face to this boundary zone
   * @param faceID ID of the face to add
   */
  void addFace(MeshFaceID faceID) { m_boundaryFaces.push_back(faceID); }

  /**
   * @brief Add multiple faces to this boundary zone
   * @param faceIDs Vector of face IDs to add
   */
  void addFaces(const std::vector<MeshFaceID> &faceIDs) {
    m_boundaryFaces.insert(m_boundaryFaces.end(), faceIDs.begin(),
                           faceIDs.end());
  }

  /**
   * @brief Get all faces in this boundary zone
   * @return Vector of face IDs
   */
  const std::vector<MeshFaceID> &getBoundaryFaces() const {
    return m_boundaryFaces;
  }

  /**
   * @brief Get the boundary condition for this zone
   * @return Shared pointer to the boundary condition
   */
  std::shared_ptr<BoundaryClass> getBoundaryCondition() const {
    return m_boundaryCondition;
  }

  /**
   * @brief Set a new boundary condition for this zone
   * @param bc Shared pointer to the new boundary condition
   */
  void setBoundaryCondition(std::shared_ptr<BoundaryClass> bc) {
    m_boundaryCondition = bc;
  }

  /**
   * @brief Check if this boundary zone is active
   * @return True if the zone is active, false otherwise
   */
  bool isActive() const { return m_isActive; }

  /**
   * @brief Set the active state of this boundary zone
   * @param active New active state
   */
  void setActive(bool active) { m_isActive = active; }

  /**
   * @brief Get the number of faces in this boundary zone
   * @return Number of faces
   */
  size_t size() const { return m_boundaryFaces.size(); }

  /**
   * @brief Check if this boundary zone contains any faces
   * @return True if the zone is empty, false otherwise
   */
  bool empty() const { return m_boundaryFaces.empty(); }

  /**
   * @brief Clear all faces from this boundary zone
   */
  void clear() { m_boundaryFaces.clear(); }

  /**
   * @brief Apply the boundary condition to all faces in this zone
   * @param mesh Reference to the mesh
   * @param time Current simulation time
   * @param dt Time step size
   */
  void applyBoundaryCondition(Mesh &mesh, double time, double dt);
};