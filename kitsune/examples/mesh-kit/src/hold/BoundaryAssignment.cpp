#include "BoundaryAssignment.h"
#include "Face.h"
#include <cmath>

namespace BoundaryAssignment {

std::shared_ptr<BoundaryZone>
assignByCoordinates(Mesh &mesh, std::shared_ptr<BoundaryClass> bc,
                    const std::function<bool(double x, double y)> &selector,
                    const std::string &zoneName) {

  // Create a name for the zone if not provided
  std::string name = zoneName.empty() ? bc->getName() + "_zone" : zoneName;

  // Create the boundary zone
  auto zone = std::make_shared<BoundaryZone>(name, bc);

  // Get all boundary faces from the mesh
  auto boundaryFaces = mesh.getBoundaryFaces();

  // Select faces that match the criteria
  for (auto faceID : boundaryFaces) {
    Face &face = mesh.getFace(faceID);
    Vector2D centroid = face.getCentroid();

    if (selector(centroid.x, centroid.y)) {
      zone->addFace(faceID);
    }
  }

  // Add the zone to the mesh
  mesh.addBoundaryZone(zone);

  return zone;
}

std::shared_ptr<BoundaryZone>
assignByRegionName(Mesh &mesh, std::shared_ptr<BoundaryClass> bc,
                   const std::string &regionName, const std::string &zoneName) {

  // Create a name for the zone if not provided
  std::string name =
      zoneName.empty() ? regionName + "_" + bc->getName() : zoneName;

  // Create the boundary zone
  auto zone = std::make_shared<BoundaryZone>(name, bc);

  // Get faces from the named region
  auto faces = mesh.getBoundaryFacesByRegion(regionName);

  // Add all faces to the zone
  zone->addFaces(faces);

  // Add the zone to the mesh
  mesh.addBoundaryZone(zone);

  return zone;
}

std::shared_ptr<BoundaryZone>
assignByNormalDirection(Mesh &mesh, std::shared_ptr<BoundaryClass> bc,
                        const Vector2D &direction, double angleTolerance,
                        const std::string &zoneName) {

  // Create a name for the zone if not provided
  std::string name =
      zoneName.empty() ? bc->getName() + "_normal_zone" : zoneName;

  // Create the boundary zone
  auto zone = std::make_shared<BoundaryZone>(name, bc);

  // Normalize the direction vector
  Vector2D normalizedDirection = direction.normalized();

  // Get all boundary faces from the mesh
  auto boundaryFaces = mesh.getBoundaryFaces();

  // Select faces with normals aligned with the specified direction
  for (auto faceID : boundaryFaces) {
    Vector2D normal = mesh.getFaceNormal(faceID);
    double dotProduct = normal.dot(normalizedDirection);

    // Check if the angle between normal and direction is within tolerance
    if (dotProduct >= angleTolerance) {
      zone->addFace(faceID);
    }
  }

  // Add the zone to the mesh
  mesh.addBoundaryZone(zone);

  return zone;
}

std::shared_ptr<BoundaryZone> assignByPredicate(
    Mesh &mesh, std::shared_ptr<BoundaryClass> bc,
    const std::function<bool(MeshFaceID, const Face &)> &predicate,
    const std::string &zoneName) {

  // Create a name for the zone if not provided
  std::string name =
      zoneName.empty() ? bc->getName() + "_custom_zone" : zoneName;

  // Create the boundary zone
  auto zone = std::make_shared<BoundaryZone>(name, bc);

  // Get all boundary faces from the mesh
  auto boundaryFaces = mesh.getBoundaryFaces();

  // Select faces that match the predicate
  for (auto faceID : boundaryFaces) {
    Face &face = mesh.getFace(faceID);

    if (predicate(faceID, face)) {
      zone->addFace(faceID);
    }
  }

  // Add the zone to the mesh
  mesh.addBoundaryZone(zone);

  return zone;
}

namespace Helpers {

std::function<bool(double x, double y)> leftBoundary(double xMin,
                                                     double tolerance) {
  return [xMin, tolerance](double x, double y) -> bool {
    return std::abs(x - xMin) <= tolerance;
  };
}

std::function<bool(double x, double y)> rightBoundary(double xMax,
                                                      double tolerance) {
  return [xMax, tolerance](double x, double y) -> bool {
    return std::abs(x - xMax) <= tolerance;
  };
}

std::function<bool(double x, double y)> bottomBoundary(double yMin,
                                                       double tolerance) {
  return [yMin, tolerance](double x, double y) -> bool {
    return std::abs(y - yMin) <= tolerance;
  };
}

std::function<bool(double x, double y)> topBoundary(double yMax,
                                                    double tolerance) {
  return [yMax, tolerance](double x, double y) -> bool {
    return std::abs(y - yMax) <= tolerance;
  };
}

std::function<bool(double x, double y)> circularBoundary(double centerX,
                                                         double centerY,
                                                         double radius,
                                                         double tolerance) {

  return [centerX, centerY, radius, tolerance](double x, double y) -> bool {
    double dx = x - centerX;
    double dy = y - centerY;
    double distance = std::sqrt(dx * dx + dy * dy);
    return std::abs(distance - radius) <= tolerance;
  };
}

} // namespace Helpers

} // namespace BoundaryAssignment