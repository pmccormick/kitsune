#pragma once

#include "Mesh.h"
#include "Cell.h"
#include "Material.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <functional>

/**
 * @file MeshObstacleUtils.h
 * @brief Utility functions for creating various obstacle shapes in a 2D mesh
 * 
 * This file provides standalone functions for creating geometric obstacles
 * such as rectangles, circles, ellipses, triangles, and polygons in a mesh.
 * Each function converts from physical coordinates to grid indices and sets
 * the appropriate cells as obstacles with the specified material.
 */

namespace MeshUtils {

/**
 * @brief Create a rectangular obstacle in the mesh
 * 
 * @param mesh Reference to the mesh to operate on
 * @param x1 Physical x-coordinate of the lower-left corner
 * @param y1 Physical y-coordinate of the lower-left corner
 * @param x2 Physical x-coordinate of the upper-right corner
 * @param y2 Physical y-coordinate of the upper-right corner
 * @param material Material to assign to the obstacle cells (nullptr for default)
 * @param fillInterior Whether to fill the interior of the obstacle (true) or just the boundary (false)
 */
inline void createRectangularObstacle(Mesh& mesh, 
                               double x1, double y1, double x2, double y2,
                               std::shared_ptr<Material> material = nullptr,
                               bool fillInterior = true) {
    // Ensure coordinates are in the correct order
    if (x1 > x2) std::swap(x1, x2);
    if (y1 > y2) std::swap(y1, y2);
    
    // Convert physical coordinates to grid indices
    size_t iStart = mesh.gridI(x1);
    size_t iEnd = mesh.gridI(x2);
    size_t jStart = mesh.gridJ(y1);
    size_t jEnd = mesh.gridJ(y2);
    
    // Ensure indices are within grid bounds
    iStart = std::max(size_t(0), std::min(iStart, mesh.getNx() - 1));
    iEnd = std::max(size_t(0), std::min(iEnd, mesh.getNx() - 1));
    jStart = std::max(size_t(0), std::min(jStart, mesh.getNy() - 1));
    jEnd = std::max(size_t(0), std::min(jEnd, mesh.getNy() - 1));
    
    // Set cells as obstacles
    for (size_t j = jStart; j <= jEnd; j++) {
        for (size_t i = iStart; i <= iEnd; i++) {
            // If fillInterior is false, only set cells at the boundary
            if (fillInterior || i == iStart || i == iEnd || j == jStart || j == jEnd) {
                mesh.setCellAsObstacle(i, j, material);
            }
        }
    }
}

/**
 * @brief Create a circular obstacle in the mesh
 * 
 * @param mesh Reference to the mesh to operate on
 * @param centerX Physical x-coordinate of the circle center
 * @param centerY Physical y-coordinate of the circle center
 * @param radius Physical radius of the circle
 * @param material Material to assign to the obstacle cells (nullptr for default)
 * @param fillInterior Whether to fill the interior of the obstacle (true) or just the boundary (false)
 */
inline void createCircularObstacle(Mesh& mesh, 
                            double centerX, double centerY, double radius,
                            std::shared_ptr<Material> material = nullptr,
                            bool fillInterior = true) {
    // Convert center coordinates to grid coordinates
    double centerI = (centerX - mesh.getOriginX()) / mesh.getDx();
    double centerJ = (centerY - mesh.getOriginY()) / mesh.getDy();
    
    // Convert radius to grid units
    double radiusI = radius / mesh.getDx();
    double radiusJ = radius / mesh.getDy();
    
    // Define bounding box in grid coordinates
    size_t iStart = std::max(0, static_cast<int>(centerI - radiusI - 1));
    size_t iEnd = std::min(mesh.getNx() - 1, static_cast<size_t>(centerI + radiusI + 1));
    size_t jStart = std::max(0, static_cast<int>(centerJ - radiusJ - 1));
    size_t jEnd = std::min(mesh.getNy() - 1, static_cast<size_t>(centerJ + radiusJ + 1));
    
    // Squared radius for distance check (avoids sqrt)
    double radiusSquared = radius * radius;
    
    // Check each cell in the bounding box
    for (size_t j = jStart; j <= jEnd; j++) {
        for (size_t i = iStart; i <= iEnd; i++) {
            // Get physical coordinates of cell center
            double x = mesh.physicalX(i);
            double y = mesh.physicalY(j);
            
            // Calculate distance from cell center to circle center
            double dx = x - centerX;
            double dy = y - centerY;
            double distanceSquared = dx*dx + dy*dy;
            
            // Check if cell is inside the circle
            if (distanceSquared <= radiusSquared) {
                // If not filling interior, check if it's a boundary cell
                if (fillInterior) {
                    mesh.setCellAsObstacle(i, j, material);
                } else {
                    // For unfilled circles, we need a way to detect cells near the boundary
                    // One approach: check if any neighboring cell is outside the circle
                    bool isBoundary = false;
                    
                    // Check in 8 directions
                    const int di[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
                    const int dj[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
                    
                    for (int n = 0; n < 8; n++) {
                        int ni = static_cast<int>(i) + di[n];
                        int nj = static_cast<int>(j) + dj[n];
                        
                        // Skip if outside grid
                        if (ni < 0 || ni >= static_cast<int>(mesh.getNx()) || 
                            nj < 0 || nj >= static_cast<int>(mesh.getNy())) 
                            continue;
                        
                        // Get physical coordinates of neighbor
                        double nx = mesh.physicalX(ni);
                        double ny = mesh.physicalY(nj);
                        
                        // Calculate distance of neighbor to circle center
                        double ndx = nx - centerX;
                        double ndy = ny - centerY;
                        double neighborDistSquared = ndx*ndx + ndy*ndy;
                        
                        // If neighbor is outside, this is a boundary cell
                        if (neighborDistSquared > radiusSquared) {
                            isBoundary = true;
                            break;
                        }
                    }
                    
                    if (isBoundary) {
                        mesh.setCellAsObstacle(i, j, material);
                    }
                }
            }
        }
    }
}

/**
 * @brief Create an elliptical obstacle in the mesh
 * 
 * @param mesh Reference to the mesh to operate on
 * @param centerX Physical x-coordinate of the ellipse center
 * @param centerY Physical y-coordinate of the ellipse center
 * @param radiusX Semi-major axis in x-direction
 * @param radiusY Semi-major axis in y-direction
 * @param rotationAngle Rotation angle in radians
 * @param material Material to assign to the obstacle cells (nullptr for default)
 * @param fillInterior Whether to fill the interior of the obstacle (true) or just the boundary (false)
 */
inline void createEllipticalObstacle(Mesh& mesh, 
                              double centerX, double centerY, 
                              double radiusX, double radiusY, double rotationAngle,
                              std::shared_ptr<Material> material = nullptr,
                              bool fillInterior = true) {
    // Precompute sin and cos of rotation angle
    double cosTheta = cos(rotationAngle);
    double sinTheta = sin(rotationAngle);
    
    // Convert center coordinates to grid coordinates
    double centerI = (centerX - mesh.getOriginX()) / mesh.getDx();
    double centerJ = (centerY - mesh.getOriginY()) / mesh.getDy();
    
    // Convert radii to grid units (approximate for rotated ellipse)
    double maxRadius = std::max(radiusX, radiusY);
    double gridRadius = maxRadius / std::min(mesh.getDx(), mesh.getDy());
    
    // Define bounding box in grid coordinates (conservative estimate)
    size_t iStart = std::max(0, static_cast<int>(centerI - gridRadius - 1));
    size_t iEnd = std::min(mesh.getNx() - 1, static_cast<size_t>(centerI + gridRadius + 1));
    size_t jStart = std::max(0, static_cast<int>(centerJ - gridRadius - 1));
    size_t jEnd = std::min(mesh.getNy() - 1, static_cast<size_t>(centerJ + gridRadius + 1));
    
    // Check each cell in the bounding box
    for (size_t j = jStart; j <= jEnd; j++) {
        for (size_t i = iStart; i <= iEnd; i++) {
            // Get physical coordinates of cell center
            double x = mesh.physicalX(i);
            double y = mesh.physicalY(j);
            
            // Translate relative to ellipse center
            double dx = x - centerX;
            double dy = y - centerY;
            
            // Rotate to align with ellipse axes
            double rotatedX = dx * cosTheta + dy * sinTheta;
            double rotatedY = -dx * sinTheta + dy * cosTheta;
            
            // Check if point is inside the ellipse: (x/a)² + (y/b)² <= 1
            double ellipseValue = (rotatedX*rotatedX)/(radiusX*radiusX) + 
                                  (rotatedY*rotatedY)/(radiusY*radiusY);
            
            if (ellipseValue <= 1.0) {
                // If not filling interior, determine if it's a boundary cell
                if (fillInterior) {
                    mesh.setCellAsObstacle(i, j, material);
                } else {
                    // For unfilled ellipses, check neighbors similar to circle case
                    bool isBoundary = false;
                    
                    // Check in 8 directions
                    const int di[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
                    const int dj[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
                    
                    for (int n = 0; n < 8; n++) {
                        int ni = static_cast<int>(i) + di[n];
                        int nj = static_cast<int>(j) + dj[n];
                        
                        // Skip if outside grid
                        if (ni < 0 || ni >= static_cast<int>(mesh.getNx()) || 
                            nj < 0 || nj >= static_cast<int>(mesh.getNy())) 
                            continue;
                        
                        // Get physical coordinates of neighbor
                        double nx = mesh.physicalX(ni);
                        double ny = mesh.physicalY(nj);
                        
                        // Translate and rotate
                        double ndx = nx - centerX;
                        double ndy = ny - centerY;
                        double nRotatedX = ndx * cosTheta + ndy * sinTheta;
                        double nRotatedY = -ndx * sinTheta + ndy * cosTheta;
                        
                        // Check if neighbor is outside ellipse
                        double nEllipseValue = (nRotatedX*nRotatedX)/(radiusX*radiusX) + 
                                              (nRotatedY*nRotatedY)/(radiusY*radiusY);
                        
                        if (nEllipseValue > 1.0) {
                            isBoundary = true;
                            break;
                        }
                    }
                    
                    if (isBoundary) {
                        mesh.setCellAsObstacle(i, j, material);
                    }
                }
            }
        }
    }
}

/**
 * @brief Helper function to check if a point is inside a triangle
 * 
 * Uses barycentric coordinates to determine if a point is inside a triangle.
 * 
 * @param x X-coordinate of the point to check
 * @param y Y-coordinate of the point to check
 * @param x1 X-coordinate of first triangle vertex
 * @param y1 Y-coordinate of first triangle vertex
 * @param x2 X-coordinate of second triangle vertex
 * @param y2 Y-coordinate of second triangle vertex
 * @param x3 X-coordinate of third triangle vertex
 * @param y3 Y-coordinate of third triangle vertex
 * @return true if the point is inside the triangle, false otherwise
 */
inline bool isPointInTriangle(double x, double y, 
                       double x1, double y1, 
                       double x2, double y2, 
                       double x3, double y3) {
    // Compute barycentric coordinates
    double denominator = ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3));
    
    // Handle degenerate triangles
    if (std::abs(denominator) < 1e-10) {
        return false;
    }
    
    double a = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denominator;
    double b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denominator;
    double c = 1 - a - b;
    
    // Check if point is inside triangle (including on the edges)
    return (a >= 0) && (b >= 0) && (c >= 0);
}

/**
 * @brief Compute the distance from a point to a line segment
 * 
 * @param x X-coordinate of the point
 * @param y Y-coordinate of the point
 * @param x1 X-coordinate of first line segment endpoint
 * @param y1 Y-coordinate of first line segment endpoint
 * @param x2 X-coordinate of second line segment endpoint
 * @param y2 Y-coordinate of second line segment endpoint
 * @return Distance from the point to the line segment
 */
inline double distanceToLineSegment(double x, double y, 
                             double x1, double y1, 
                             double x2, double y2) {
    // Calculate squared length of line segment
    double lenSquared = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1);
    
    // If segment is a point, return distance to that point
    if (lenSquared < 1e-10) {
        return std::sqrt((x - x1) * (x - x1) + (y - y1) * (y - y1));
    }
    
    // Compute projection parameter (t)
    double t = std::max(0.0, std::min(1.0, 
        ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / lenSquared));
    
    // Compute closest point on segment
    double projX = x1 + t * (x2 - x1);
    double projY = y1 + t * (y2 - y1);
    
    // Return distance to closest point
    return std::sqrt((x - projX) * (x - projX) + (y - projY) * (y - projY));
}

/**
 * @brief Create a triangular obstacle in the mesh
 * 
 * @param mesh Reference to the mesh to operate on
 * @param x1 Physical x-coordinate of first vertex
 * @param y1 Physical y-coordinate of first vertex
 * @param x2 Physical x-coordinate of second vertex
 * @param y2 Physical y-coordinate of second vertex
 * @param x3 Physical x-coordinate of third vertex
 * @param y3 Physical y-coordinate of third vertex
 * @param material Material to assign to the obstacle cells (nullptr for default)
 * @param fillInterior Whether to fill the interior of the obstacle (true) or just the boundary (false)
 */
inline void createTriangularObstacle(Mesh& mesh,
                              double x1, double y1, 
                              double x2, double y2, 
                              double x3, double y3,
                              std::shared_ptr<Material> material = nullptr,
                              bool fillInterior = true) {
    // Find bounding box of the triangle
    double minX = std::min({x1, x2, x3});
    double maxX = std::max({x1, x2, x3});
    double minY = std::min({y1, y2, y3});
    double maxY = std::max({y1, y2, y3});
    
    // Convert to grid indices with padding
    size_t iStart = std::max(0, static_cast<int>(mesh.gridI(minX) - 1));
    size_t iEnd = std::min(mesh.getNx() - 1, mesh.gridI(maxX) + 1);
    size_t jStart = std::max(0, static_cast<int>(mesh.gridJ(minY) - 1));
    size_t jEnd = std::min(mesh.getNy() - 1, mesh.gridJ(maxY) + 1);
    
    // Constant for boundary thickness (in physical units)
    double boundaryThickness = 0.5 * std::min(mesh.getDx(), mesh.getDy());
    
    // Check each cell in the bounding box
    for (size_t j = jStart; j <= jEnd; j++) {
        for (size_t i = iStart; i <= iEnd; i++) {
            // Get physical coordinates of cell center
            double x = mesh.physicalX(i);
            double y = mesh.physicalY(j);
            
            bool isInside = isPointInTriangle(x, y, x1, y1, x2, y2, x3, y3);
            
            if (isInside) {
                // For filled triangles, mark all interior cells
                if (fillInterior) {
                    mesh.setCellAsObstacle(i, j, material);
                } else {
                    // For unfilled triangles, compute distance to each edge
                    double dist1 = distanceToLineSegment(x, y, x1, y1, x2, y2);
                    double dist2 = distanceToLineSegment(x, y, x2, y2, x3, y3);
                    double dist3 = distanceToLineSegment(x, y, x3, y3, x1, y1);
                    
                    // Minimum distance to any edge
                    double minDist = std::min({dist1, dist2, dist3});
                    
                    // If close to any edge, mark as boundary
                    if (minDist <= boundaryThickness) {
                        mesh.setCellAsObstacle(i, j, material);
                    }
                }
            }
        }
    }
}

/**
 * @brief Create a polygonal obstacle in the mesh
 * 
 * @param mesh Reference to the mesh to operate on
 * @param vertices Vector of (x,y) coordinates forming the polygon vertices in counter-clockwise order
 * @param material Material to assign to the obstacle cells (nullptr for default)
 * @param fillInterior Whether to fill the interior of the obstacle (true) or just the boundary (false)
 */
inline void createPolygonalObstacle(Mesh& mesh,
                             const std::vector<std::pair<double, double>>& vertices,
                             std::shared_ptr<Material> material = nullptr,
                             bool fillInterior = true) {
    // Need at least 3 vertices for a polygon
    if (vertices.size() < 3) return;
    
    // Find bounding box of the polygon
    double minX = vertices[0].first;
    double maxX = vertices[0].first;
    double minY = vertices[0].second;
    double maxY = vertices[0].second;
    
    for (const auto& vertex : vertices) {
        minX = std::min(minX, vertex.first);
        maxX = std::max(maxX, vertex.first);
        minY = std::min(minY, vertex.second);
        maxY = std::max(maxY, vertex.second);
    }
    
    // Convert to grid indices with padding
    size_t iStart = std::max(0, static_cast<int>(mesh.gridI(minX) - 1));
    size_t iEnd = std::min(mesh.getNx() - 1, mesh.gridI(maxX) + 1);
    size_t jStart = std::max(0, static_cast<int>(mesh.gridJ(minY) - 1));
    size_t jEnd = std::min(mesh.getNy() - 1, mesh.gridJ(maxY) + 1);
    
    // Constant for boundary thickness (in physical units)
    double boundaryThickness = 0.5 * std::min(mesh.getDx(), mesh.getDy());
    
    // Check each cell in the bounding box
    for (size_t j = jStart; j <= jEnd; j++) {
        for (size_t i = iStart; i <= iEnd; i++) {
            // Get physical coordinates of cell center
            double x = mesh.physicalX(i);
            double y = mesh.physicalY(j);
            
            // Point-in-polygon test using ray casting algorithm
            bool isInside = false;
            size_t vertexCount = vertices.size();
            
            for (size_t v = 0, u = vertexCount - 1; v < vertexCount; u = v++) {
                const auto& current = vertices[v];
                const auto& previous = vertices[u];
                
                // Check if ray from point crosses this edge
                if (((current.second > y) != (previous.second > y)) && 
                    (x < (previous.first - current.first) * (y - current.second) / 
                     (previous.second - current.second) + current.first)) {
                    isInside = !isInside;
                }
            }
            
            if (isInside) {
                // For filled polygons, mark all interior cells
                if (fillInterior) {
                    mesh.setCellAsObstacle(i, j, material);
                } else {
                    // For unfilled polygons, compute distance to each edge
                    double minDist = std::numeric_limits<double>::max();
                    
                    for (size_t v = 0, u = vertexCount - 1; v < vertexCount; u = v++) {
                        const auto& current = vertices[v];
                        const auto& previous = vertices[u];
                        
                        double dist = distanceToLineSegment(
                            x, y, previous.first, previous.second, current.first, current.second);
                        
                        minDist = std::min(minDist, dist);
                    }
                    
                    // If close to any edge, mark as boundary
                    if (minDist <= boundaryThickness) {
                        mesh.setCellAsObstacle(i, j, material);
                    }
                }
            }
        }
    }
}

} // namespace MeshUtils
 
