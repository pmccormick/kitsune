#pragma once

#include "BoundaryClass.h"
#include "BoundaryZone.h"
#include "Mesh.h"
#include "Vector2D.h"
#include <functional>
#include <memory>
#include <string>
#include <vector>

/**
 * @namespace BoundaryAssignment
 * @brief Contains functions for assigning boundary conditions to mesh regions
 */
namespace BoundaryAssignment {

/**
 * @brief Assign a boundary condition to faces matching geometric criteria
 * @param mesh Reference to the mesh
 * @param bc Shared pointer to the boundary condition
 * @param selector Function that selects faces based on coordinates
 * @param zoneName Optional name for the boundary zone (default uses boundary
 * name)
 * @return Shared pointer to the created boundary zone
 */
std::shared_ptr<BoundaryZone>
assignByCoordinates(Mesh &mesh, std::shared_ptr<BoundaryClass> bc,
                    const std::function<bool(double x, double y)> &selector,
                    const std::string &zoneName = "");

/**
 * @brief Assign a boundary condition to faces in a named region
 * @param mesh Reference to the mesh
 * @param bc Shared pointer to the boundary condition
 * @param regionName Name of the boundary region in the mesh
 * @param zoneName Optional name for the boundary zone (default uses region
 * name)
 * @return Shared pointer to the created boundary zone
 */
std::shared_ptr<BoundaryZone>
assignByRegionName(Mesh &mesh, std::shared_ptr<BoundaryClass> bc,
                   const std::string &regionName,
                   const std::string &zoneName = "");

/**
 * @brief Assign a boundary condition to faces with normal vectors in a specific
 * direction
 * @param mesh Reference to the mesh
 * @param bc Shared pointer to the boundary condition
 * @param direction Reference direction vector
 * @param angleTolerance Cosine of the maximum angle between face normal and
 * direction
 * @param zoneName Optional name for the boundary zone (default uses boundary
 * name)
 * @return Shared pointer to the created boundary zone
 */
std::shared_ptr<BoundaryZone> assignByNormalDirection(
    Mesh &mesh, std::shared_ptr<BoundaryClass> bc, const Vector2D &direction,
    double angleTolerance = 0.95, // Default to about 18 degrees
    const std::string &zoneName = "");

/**
 * @brief Assign a boundary condition to faces matching a custom predicate
 * @param mesh Reference to the mesh
 * @param bc Shared pointer to the boundary condition
 * @param predicate Function that evaluates whether to include a face
 * @param zoneName Optional name for the boundary zone (default uses boundary
 * name)
 * @return Shared pointer to the created boundary zone
 */
std::shared_ptr<BoundaryZone> assignByPredicate(
    Mesh &mesh, std::shared_ptr<BoundaryClass> bc,
    const std::function<bool(MeshFaceID, const Face &)> &predicate,
    const std::string &zoneName = "");

/**
 * @brief Create helpers for common boundary configurations
 */
namespace Helpers {
/**
 * @brief Create a selector for the left boundary of a rectangular domain
 * @param xMin Minimum x-coordinate of the domain
 * @param tolerance Distance tolerance
 * @return Selector function
 */
std::function<bool(double x, double y)> leftBoundary(double xMin,
                                                     double tolerance = 1e-6);

/**
 * @brief Create a selector for the right boundary of a rectangular domain
 * @param xMax Maximum x-coordinate of the domain
 * @param tolerance Distance tolerance
 * @return Selector function
 */
std::function<bool(double x, double y)> rightBoundary(double xMax,
                                                      double tolerance = 1e-6);

/**
 * @brief Create a selector for the bottom boundary of a rectangular domain
 * @param yMin Minimum y-coordinate of the domain
 * @param tolerance Distance tolerance
 * @return Selector function
 */
std::function<bool(double x, double y)> bottomBoundary(double yMin,
                                                       double tolerance = 1e-6);

/**
 * @brief Create a selector for the top boundary of a rectangular domain
 * @param yMax Maximum y-coordinate of the domain
 * @param tolerance Distance tolerance
 * @return Selector function
 */
std::function<bool(double x, double y)> topBoundary(double yMax,
                                                    double tolerance = 1e-6);

/**
 * @brief Create a selector for a circular boundary
 * @param centerX X-coordinate of the circle center
 * @param centerY Y-coordinate of the circle center
 * @param radius Radius of the circle
 * @param tolerance Distance tolerance
 * @return Selector function
 */
std::function<bool(double x, double y)>
circularBoundary(double centerX, double centerY, double radius,
                 double tolerance = 1e-6);
} // namespace Helpers

} // namespace BoundaryAssignment