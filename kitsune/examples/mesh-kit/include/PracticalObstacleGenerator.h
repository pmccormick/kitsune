#pragma once

#include "Mesh.h"
#include "Cell.h"
#include "Material.h"
#include "MeshObstacleUtils.h" // Include basic shapes for reuse
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <sstream>
#include <stdexcept>

/**
 * @file MeshFlowObstacleUtils.h
 * @brief Utility functions for creating practical CFD test case obstacles
 * 
 * This file provides standalone functions for creating various practical
 * flow obstacles such as airfoils, steps, cavities, cylinders, and bluff bodies.
 * These shapes are commonly used in CFD test cases and benchmarks.
 */

namespace MeshUtils {

/**
 * @brief Generate points for a NACA 4-digit airfoil
 * 
 * @param naca4Code NACA 4-digit code (e.g., "0012", "2412")
 * @param numPoints Number of points to generate along the profile
 * @return Vector of (x,y) coordinate pairs defining the airfoil shape from trailing to leading edge
 */
inline std::vector<std::pair<double, double>> generateNACA4DigitProfile(
        const std::string& naca4Code, 
        size_t numPoints = 100) {
    
    // Ensure valid NACA 4-digit code
    if (naca4Code.length() != 4 || !std::all_of(naca4Code.begin(), naca4Code.end(), ::isdigit)) {
        throw std::invalid_argument("Invalid NACA code: " + naca4Code + ". Must be 4 digits.");
    }
    
    // Parse NACA parameters
    double m = static_cast<double>(naca4Code[0] - '0') / 100.0;  // Maximum camber
    double p = static_cast<double>(naca4Code[1] - '0') / 10.0;   // Location of maximum camber
    double t = static_cast<double>((naca4Code[2] - '0') * 10 + (naca4Code[3] - '0')) / 100.0;  // Thickness
    
    // If p is zero and we need camber, set it to a small value to avoid division by zero
    if (p == 0.0 && m > 0.0) p = 0.001;
    
    // Generate airfoil points
    std::vector<std::pair<double, double>> points;
    points.reserve(numPoints);
    
    // Cosine spacing for better resolution near leading edge
    for (size_t i = 0; i < numPoints / 2; i++) {
        double beta = M_PI * static_cast<double>(i) / (numPoints / 2 - 1);
        double x = 0.5 * (1.0 - cos(beta));  // Ranges from 0 to 1
        
        // Calculate thickness distribution
        double yt = t / 0.2 * (0.2969 * sqrt(x) - 0.1260 * x - 0.3516 * x * x + 0.2843 * pow(x, 3) - 0.1015 * pow(x, 4));
        
        // Calculate camber and local angle
        double yc = 0.0;
        double dyc_dx = 0.0;
        
        if (m > 0.0) {  // Only calculate camber if m > 0
            if (x < p) {
                yc = m * (x / p) * (2.0 * p - x / p);
                dyc_dx = 2.0 * m * (p - x) / (p * p);
            } else {
                yc = m * (1.0 - 2.0 * p + 2.0 * p * x - x * x) / ((1.0 - p) * (1.0 - p));
                dyc_dx = 2.0 * m * (p - x) / ((1.0 - p) * (1.0 - p));
            }
        }
        
        // Angle and points on upper and lower surfaces
        double theta = atan(dyc_dx);
        double xu = x - yt * sin(theta);
        double yu = yc + yt * cos(theta);
        double xl = x + yt * sin(theta);
        double yl = yc - yt * cos(theta);
        
        // Add points (starting from trailing edge, going around to trailing edge again)
        // Upper surface (from trailing to leading edge)
        points.emplace_back(1.0 - xu, yu);
    }
    
    // Lower surface (from leading to trailing edge)
    for (size_t i = 0; i < numPoints / 2; i++) {
        double beta = M_PI * static_cast<double>(i) / (numPoints / 2 - 1);
        double x = 0.5 * (1.0 - cos(beta));  // Ranges from 0 to 1
        
        // Calculate thickness distribution
        double yt = t / 0.2 * (0.2969 * sqrt(x) - 0.1260 * x - 0.3516 * x * x + 0.2843 * pow(x, 3) - 0.1015 * pow(x, 4));
        
        // Calculate camber and local angle
        double yc = 0.0;
        double dyc_dx = 0.0;
        
        if (m > 0.0) {  // Only calculate camber if m > 0
            if (x < p) {
                yc = m * (x / p) * (2.0 * p - x / p);
                dyc_dx = 2.0 * m * (p - x) / (p * p);
            } else {
                yc = m * (1.0 - 2.0 * p + 2.0 * p * x - x * x) / ((1.0 - p) * (1.0 - p));
                dyc_dx = 2.0 * m * (p - x) / ((1.0 - p) * (1.0 - p));
            }
        }
        
        // Angle and points on upper and lower surfaces
        double theta = atan(dyc_dx);
        double xl = x + yt * sin(theta);
        double yl = yc - yt * cos(theta);
        
        // Add lower surface point (from leading to trailing edge)
        points.emplace_back(xl, yl);
    }
    
    return points;
}

/**
 * @brief Create an airfoil obstacle in the mesh
 * 
 * @param mesh Reference to the mesh to operate on
 * @param leadingEdgeX Physical x-coordinate of the airfoil leading edge
 * @param leadingEdgeY Physical y-coordinate of the airfoil leading edge
 * @param chordLength Length of the airfoil chord
 * @param angle Angle of attack in radians
 * @param airfoilType NACA code (e.g., "0012", "2412") or other profile type
 * @param material Material to assign to the obstacle cells (nullptr for default)
 * @param fillInterior Whether to fill the interior of the obstacle (true) or just the boundary (false)
 */
inline void createAirfoilObstacle(Mesh& mesh, 
                          double leadingEdgeX, double leadingEdgeY,
                          double chordLength, double angle,
                          const std::string& airfoilType = "NACA0012",
                          std::shared_ptr<Material> material = nullptr,
                          bool fillInterior = true) {
    
    // Handle different airfoil types
    std::vector<std::pair<double, double>> airfoilPoints;
    
    // Parse the airfoil type
    if (airfoilType.substr(0, 4) == "NACA") {
        // Extract the NACA digits
        std::string nacaCode = airfoilType.substr(4);
        if (nacaCode.length() == 4 && std::all_of(nacaCode.begin(), nacaCode.end(), ::isdigit)) {
            // 4-digit NACA airfoil
            airfoilPoints = generateNACA4DigitProfile(nacaCode, 200);
        } else {
            // Default to NACA0012 if invalid code
            airfoilPoints = generateNACA4DigitProfile("0012", 200);
        }
    } else {
        // Default to NACA0012 for unrecognized types
        airfoilPoints = generateNACA4DigitProfile("0012", 200);
    }
    
    // Scale, rotate, and translate the airfoil points
    std::vector<std::pair<double, double>> transformedPoints;
    transformedPoints.reserve(airfoilPoints.size());
    
    double cosAngle = cos(angle);
    double sinAngle = sin(angle);
    
    for (const auto& point : airfoilPoints) {
        // Scale by chord length
        double scaledX = point.first * chordLength;
        double scaledY = point.second * chordLength;
        
        // Rotate by angle of attack
        double rotatedX = scaledX * cosAngle - scaledY * sinAngle;
        double rotatedY = scaledX * sinAngle + scaledY * cosAngle;
        
        // Translate to leading edge position
        double finalX = leadingEdgeX + rotatedX;
        double finalY = leadingEdgeY + rotatedY;
        
        transformedPoints.emplace_back(finalX, finalY);
    }
    
    // Create the airfoil as a polygonal obstacle
    createPolygonalObstacle(mesh, transformedPoints, material, fillInterior);
}

/**
 * @brief Create a backward-facing step obstacle in the mesh
 * 
 * @param mesh Reference to the mesh to operate on
 * @param stepX Physical x-coordinate of the step
 * @param stepHeight Height of the step
 * @param inletHeight Height of the inlet channel
 * @param outletLength Length of the outlet channel after the step
 * @param material Material to assign to the obstacle cells (nullptr for default)
 */
inline void createBackwardFacingStep(Mesh& mesh,
                              double stepX, double stepHeight, 
                              double inletHeight, double outletLength,
                              std::shared_ptr<Material> material = nullptr) {
    
    // Get grid dimensions and domain size
    double minX = mesh.getOriginX();
    double minY = mesh.getOriginY();
    double maxX = minX + mesh.getWidth();
    double maxY = minY + mesh.getHeight();
    
    // Ensure step is within domain
    stepX = std::max(minX, std::min(maxX, stepX));
    
    // Create bottom wall
    createRectangularObstacle(mesh, minX, minY, maxX, minY, material, true);
    
    // Create top wall
    createRectangularObstacle(mesh, minX, maxY - inletHeight, maxX, maxY, material, true);
    
    // Create step
    createRectangularObstacle(mesh, stepX, minY, stepX + outletLength, minY + stepHeight, material, true);
}

/**
 * @brief Create a forward-facing step obstacle in the mesh
 * 
 * @param mesh Reference to the mesh to operate on
 * @param stepX Physical x-coordinate of the step
 * @param stepHeight Height of the step
 * @param inletLength Length of the inlet channel before the step
 * @param outletHeight Height of the outlet channel
 * @param material Material to assign to the obstacle cells (nullptr for default)
 */
inline void createForwardFacingStep(Mesh& mesh,
                             double stepX, double stepHeight, 
                             double inletLength, double outletHeight,
                             std::shared_ptr<Material> material = nullptr) {
    
    // Get grid dimensions and domain size
    double minX = mesh.getOriginX();
    double minY = mesh.getOriginY();
    double maxX = minX + mesh.getWidth();
    double maxY = minY + mesh.getHeight();
    
    // Ensure step is within domain
    stepX = std::max(minX, std::min(maxX, stepX));
    
    // Create bottom wall
    createRectangularObstacle(mesh, minX, minY, maxX, minY, material, true);
    
    // Create top wall
    createRectangularObstacle(mesh, minX, maxY - outletHeight, maxX, maxY, material, true);
    
    // Create step
    createRectangularObstacle(mesh, minX, minY, stepX, minY + stepHeight, material, true);
}

/**
 * @brief Create a cavity in the mesh (for lid-driven cavity test case)
 * 
 * @param mesh Reference to the mesh to operate on
 * @param x1 Physical x-coordinate of the cavity's left edge
 * @param y1 Physical y-coordinate of the cavity's bottom edge
 * @param x2 Physical x-coordinate of the cavity's right edge
 * @param y2 Physical y-coordinate of the cavity's top edge
 * @param material Material to assign to the cavity walls (nullptr for default)
 */
inline void createCavity(Mesh& mesh,
                  double x1, double y1, double x2, double y2,
                  std::shared_ptr<Material> material = nullptr) {
    
    // Ensure coordinates are ordered correctly
    if (x1 > x2) std::swap(x1, x2);
    if (y1 > y2) std::swap(y1, y2);
    
    // Create the four walls of the cavity
    double wallThickness = std::min(mesh.getDx(), mesh.getDy());
    
    // Bottom wall
    createRectangularObstacle(mesh, x1, y1 - wallThickness, x2, y1, material, true);
    
    // Left wall
    createRectangularObstacle(mesh, x1 - wallThickness, y1, x1, y2, material, true);
    
    // Right wall
    createRectangularObstacle(mesh, x2, y1, x2 + wallThickness, y2, material, true);
    
    // The top (lid) is typically not an obstacle in driven cavity simulations
    // But we provide the option to create it
    if (false) { // Change to a parameter if needed
        createRectangularObstacle(mesh, x1, y2, x2, y2 + wallThickness, material, true);
    }
}

/**
 * @brief Create a cylinder obstacle in the mesh
 * 
 * This is a thin wrapper around createCircularObstacle for consistent naming
 * 
 * @param mesh Reference to the mesh to operate on
 * @param centerX Physical x-coordinate of the cylinder center
 * @param centerY Physical y-coordinate of the cylinder center
 * @param radius Physical radius of the cylinder
 * @param material Material to assign to the obstacle cells (nullptr for default)
 * @param fillInterior Whether to fill the interior of the obstacle (true) or just the boundary (false)
 */
inline void createCylinderObstacle(Mesh& mesh,
                            double centerX, double centerY, double radius,
                            std::shared_ptr<Material> material = nullptr,
                            bool fillInterior = true) {
    
    // Call the circular obstacle function
    createCircularObstacle(mesh, centerX, centerY, radius, material, fillInterior);
}

/**
 * @brief Create a bluff body obstacle (rectangular) in the mesh
 * 
 * @param mesh Reference to the mesh to operate on
 * @param centerX Physical x-coordinate of the bluff body center
 * @param centerY Physical y-coordinate of the bluff body center
 * @param width Width of the bluff body
 * @param height Height of the bluff body
 * @param angle Rotation angle in radians
 * @param material Material to assign to the obstacle cells (nullptr for default)
 * @param fillInterior Whether to fill the interior of the obstacle (true) or just the boundary (false)
 */
inline void createBluffBodyObstacle(Mesh& mesh,
                             double centerX, double centerY,
                             double width, double height, double angle,
                             std::shared_ptr<Material> material = nullptr,
                             bool fillInterior = true) {
    
    // If no rotation, use simpler rectangular obstacle
    if (std::abs(angle) < 1e-6) {
        double x1 = centerX - width / 2.0;
        double y1 = centerY - height / 2.0;
        double x2 = centerX + width / 2.0;
        double y2 = centerY + height / 2.0;
        
        createRectangularObstacle(mesh, x1, y1, x2, y2, material, fillInterior);
    } else {
        // Create a rotated rectangle using 4 points
        std::vector<std::pair<double, double>> vertices;
        vertices.reserve(4);
        
        double cosA = cos(angle);
        double sinA = sin(angle);
        
        // Calculate the four corners
        double dx = width / 2.0;
        double dy = height / 2.0;
        
        // Bottom-left
        vertices.emplace_back(
            centerX + (-dx * cosA - (-dy) * sinA),
            centerY + (-dx * sinA + (-dy) * cosA)
        );
        
        // Bottom-right
        vertices.emplace_back(
            centerX + (dx * cosA - (-dy) * sinA),
            centerY + (dx * sinA + (-dy) * cosA)
        );
        
        // Top-right
        vertices.emplace_back(
            centerX + (dx * cosA - dy * sinA),
            centerY + (dx * sinA + dy * cosA)
        );
        
        // Top-left
        vertices.emplace_back(
            centerX + (-dx * cosA - dy * sinA),
            centerY + (-dx * sinA + dy * cosA)
        );
        
        // Create the polygon
        createPolygonalObstacle(mesh, vertices, material, fillInterior);
    }
}

} // namespace MeshUtils
  //
  //
