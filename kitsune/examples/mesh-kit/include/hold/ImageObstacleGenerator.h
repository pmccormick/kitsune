#pragma once

#include "Mesh.h"
#include "Cell.h"
#include "Material.h"
#include <vector>
#include <string>
#include <functional>
#include <stdexcept>
#include <memory>
#include <cmath>
#include <algorithm>

// Define STB_IMAGE_IMPLEMENTATION in exactly one source file
// before including this header, e.g.:
// #define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h" // Assume this is available in your project

/**
 * @file MeshObstacleUtilsImage.h
 * @brief Utility functions for creating obstacles from images and parametric functions
 * 
 * This file provides standalone functions for creating obstacles from image files
 * and mathematical functions. These utilities allow for complex obstacle shapes
 * that would be difficult to define programmatically.
 */

namespace MeshUtils {

/**
 * @brief Convert obstacle definition function to a point cloud
 * 
 * @param insideFunction Function that returns true if a point is inside the obstacle
 * @param x1 Physical x-coordinate of sampling region's left edge
 * @param y1 Physical y-coordinate of sampling region's bottom edge
 * @param x2 Physical x-coordinate of sampling region's right edge
 * @param y2 Physical y-coordinate of sampling region's top edge
 * @param resolution Sampling resolution (points per unit length)
 * @return Vector of (x,y) coordinates representing points inside the obstacle
 */
inline std::vector<std::pair<double, double>> generateObstaclePointCloud(
    std::function<bool(double,double)> insideFunction,
    double x1, double y1, double x2, double y2,
    double resolution) {
    
    std::vector<std::pair<double, double>> points;
    
    // Calculate step size based on resolution
    double dx = 1.0 / resolution;
    double dy = 1.0 / resolution;
    
    // Ensure coordinates are ordered correctly
    if (x1 > x2) std::swap(x1, x2);
    if (y1 > y2) std::swap(y1, y2);
    
    // Calculate number of steps
    int stepsX = static_cast<int>(std::ceil((x2 - x1) * resolution));
    int stepsY = static_cast<int>(std::ceil((y2 - y1) * resolution));
    
    // Sample the function over the region
    for (int j = 0; j <= stepsY; j++) {
        double y = y1 + j * dy;
        for (int i = 0; i <= stepsX; i++) {
            double x = x1 + i * dx;
            
            // Check if point is inside
            if (insideFunction(x, y)) {
                points.emplace_back(x, y);
            }
        }
    }
    
    return points;
}

/**
 * @brief Create an obstacle from a parametric shape function
 * 
 * @param mesh Reference to the mesh to operate on
 * @param insideFunction Function that returns true if a point is inside the obstacle
 * @param x1 Physical x-coordinate of the region's left edge
 * @param y1 Physical y-coordinate of the region's bottom edge
 * @param x2 Physical x-coordinate of the region's right edge
 * @param y2 Physical y-coordinate of the region's top edge
 * @param resolution Sampling resolution (points per unit length)
 * @param material Material to assign to the obstacle cells (nullptr for default)
 */
inline void createParametricObstacle(Mesh& mesh,
                              std::function<bool(double,double)> insideFunction,
                              double x1, double y1, double x2, double y2,
                              double resolution,
                              std::shared_ptr<Material> material = nullptr) {
    
    // Generate the point cloud
    auto points = generateObstaclePointCloud(insideFunction, x1, y1, x2, y2, resolution);
    
    // Convert to grid coordinates
    for (const auto& point : points) {
        size_t i = mesh.gridI(point.first);
        size_t j = mesh.gridJ(point.second);
        
        // Check if within grid bounds
        if (i < mesh.getNx() && j < mesh.getNy()) {
            mesh.setCellAsObstacle(i, j, material);
        }
    }
}

/**
 * @brief Create obstacle from bitmap/image data
 * 
 * @param mesh Reference to the mesh to operate on
 * @param imageFilename Path to the image file (PNG, BMP, etc.)
 * @param x1 Physical x-coordinate of the image's left edge in the mesh
 * @param y1 Physical y-coordinate of the image's bottom edge in the mesh
 * @param x2 Physical x-coordinate of the image's right edge in the mesh
 * @param y2 Physical y-coordinate of the image's top edge in the mesh
 * @param material Material to assign to the obstacle cells (nullptr for default)
 * @param threshold Brightness threshold (0-255) for obstacle detection (for grayscale/alpha)
 * @param invertImage Whether to invert the image interpretation (dark=obstacle vs light=obstacle)
 */
inline void createObstacleFromImage(Mesh& mesh,
                             const std::string& imageFilename,
                             double x1, double y1, double x2, double y2,
                             std::shared_ptr<Material> material = nullptr,
                             int threshold = 128,
                             bool invertImage = false) {
    
    // Ensure coordinates are ordered correctly
    if (x1 > x2) std::swap(x1, x2);
    if (y1 > y2) std::swap(y1, y2);
    
    // Load the image using stb_image
    int width, height, channels;
    unsigned char* data = stbi_load(imageFilename.c_str(), &width, &height, &channels, 0);
    
    if (!data) {
        throw std::runtime_error("Failed to load image: " + imageFilename);
    }
    
    // Handle the image data
    try {
        // Calculate physical size of each pixel
        double pixelWidth = (x2 - x1) / width;
        double pixelHeight = (y2 - y1) / height;
        
        // Note: Image coordinate system typically has origin at top-left
        // We'll flip the y-coordinate to match our physical coordinate system
        
        // Process each pixel in the image
        for (int img_y = 0; img_y < height; img_y++) {
            // Flip y-coordinate so bottom of image maps to y1
            double physical_y = y2 - (img_y + 0.5) * pixelHeight;
            
            for (int img_x = 0; img_x < width; img_x++) {
                double physical_x = x1 + (img_x + 0.5) * pixelWidth;
                
                // Determine if this pixel is an obstacle
                bool isObstacle = false;
                
                // Get pixel data based on image format
                size_t pixelIndex = (img_y * width + img_x) * channels;
                
                if (channels == 1) {
                    // Grayscale image
                    isObstacle = (data[pixelIndex] > threshold);
                }
                else if (channels == 2) {
                    // Grayscale with alpha
                    isObstacle = (data[pixelIndex] > threshold && data[pixelIndex + 1] > threshold);
                }
                else if (channels == 3) {
                    // RGB image - use average of components
                    int avg = (data[pixelIndex] + data[pixelIndex + 1] + data[pixelIndex + 2]) / 3;
                    isObstacle = (avg > threshold);
                }
                else if (channels == 4) {
                    // RGBA image - check alpha and color
                    int avg = (data[pixelIndex] + data[pixelIndex + 1] + data[pixelIndex + 2]) / 3;
                    int alpha = data[pixelIndex + 3];
                    isObstacle = (avg > threshold && alpha > threshold);
                }
                
                // Apply inversion if requested
                if (invertImage) {
                    isObstacle = !isObstacle;
                }
                
                // If this pixel is an obstacle, set the corresponding cell
                if (isObstacle) {
                    // Convert to grid indices
                    size_t i = mesh.gridI(physical_x);
                    size_t j = mesh.gridJ(physical_y);
                    
                    // Check if within bounds
                    if (i < mesh.getNx() && j < mesh.getNy()) {
                        mesh.setCellAsObstacle(i, j, material);
                    }
                }
            }
        }
    }
    catch (...) {
        // Free image data on any exception
        stbi_image_free(data);
        throw; // Re-throw the exception
    }
    
    // Free image data
    stbi_image_free(data);
}

/**
 * @brief Create an obstacle from a binary matrix/grid (0s and 1s)
 * 
 * @param mesh Reference to the mesh to operate on
 * @param binaryMatrix Matrix of 0s (no obstacle) and 1s (obstacle)
 * @param x1 Physical x-coordinate of the matrix's left edge in the mesh
 * @param y1 Physical y-coordinate of the matrix's bottom edge in the mesh
 * @param x2 Physical x-coordinate of the matrix's right edge in the mesh
 * @param y2 Physical y-coordinate of the matrix's top edge in the mesh
 * @param material Material to assign to the obstacle cells (nullptr for default)
 */
inline void createObstacleFromMatrix(Mesh& mesh,
                              const std::vector<std::vector<int>>& binaryMatrix,
                              double x1, double y1, double x2, double y2,
                              std::shared_ptr<Material> material = nullptr) {
    
    // Ensure coordinates are ordered correctly
    if (x1 > x2) std::swap(x1, x2);
    if (y1 > y2) std::swap(y1, y2);
    
    // Check if matrix is empty
    if (binaryMatrix.empty() || binaryMatrix[0].empty()) {
        return;
    }
    
    // Get matrix dimensions
    size_t matrixHeight = binaryMatrix.size();
    size_t matrixWidth = binaryMatrix[0].size();
    
    // Calculate physical size of each matrix cell
    double cellWidth = (x2 - x1) / matrixWidth;
    double cellHeight = (y2 - y1) / matrixHeight;
    
    // Process each cell in the matrix
    for (size_t matrix_y = 0; matrix_y < matrixHeight; matrix_y++) {
        // Flip y-coordinate so bottom of matrix maps to y1
        double physical_y = y1 + (matrixHeight - matrix_y - 0.5) * cellHeight;
        
        for (size_t matrix_x = 0; matrix_x < matrixWidth; matrix_x++) {
            // Skip if this is not an obstacle (0)
            if (binaryMatrix[matrix_y][matrix_x] == 0) {
                continue;
            }
            
            // Calculate physical coordinates
            double physical_x = x1 + (matrix_x + 0.5) * cellWidth;
            
            // Convert to grid indices
            size_t i = mesh.gridI(physical_x);
            size_t j = mesh.gridJ(physical_y);
            
            // Check if within bounds
            if (i < mesh.getNx() && j < mesh.getNy()) {
                mesh.setCellAsObstacle(i, j, material);
            }
        }
    }
}

/**
 * @brief Load a binary matrix from a text file
 * 
 * @param filename Path to the text file
 * @return Matrix of 0s and 1s
 */
inline std::vector<std::vector<int>> loadBinaryMatrixFromFile(const std::string& filename) {
    std::vector<std::vector<int>> matrix;
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    
    std::string line;
    while (std::getline(file, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        std::vector<int> row;
        std::istringstream iss(line);
        char c;
        
        // Read each character
        while (iss.get(c)) {
            if (c == '0' || c == ' ' || c == '.') {
                row.push_back(0); // No obstacle
            }
            else if (c == '1' || c == 'X' || c == 'x' || c == '*') {
                row.push_back(1); // Obstacle
            }
            // Ignore other characters
        }
        
        // Add non-empty rows to the matrix
        if (!row.empty()) {
            matrix.push_back(row);
        }
    }
    
    return matrix;
}

/**
 * @brief Generate an obstacle from a distance function (signed distance field)
 * 
 * @param mesh Reference to the mesh to operate on
 * @param distanceFunc Function that returns negative inside obstacle, positive outside
 * @param x1 Physical x-coordinate of the region's left edge
 * @param y1 Physical y-coordinate of the region's bottom edge
 * @param x2 Physical x-coordinate of the region's right edge
 * @param y2 Physical y-coordinate of the region's top edge
 * @param material Material to assign to the obstacle cells (nullptr for default)
 */
inline void createObstacleFromDistanceFunction(Mesh& mesh,
                                       std::function<double(double,double)> distanceFunc,
                                       double x1, double y1, double x2, double y2,
                                       std::shared_ptr<Material> material = nullptr) {
    
    // Create a lambda that returns true for inside points
    auto insideFunc = [distanceFunc](double x, double y) -> bool {
        return distanceFunc(x, y) <= 0.0;
    };
    
    // Use the parametric obstacle function
    createParametricObstacle(mesh, insideFunc, x1, y1, x2, y2, 
                            1.0 / std::min(mesh.getDx(), mesh.getDy()), material);
}

/**
 * @brief Create an obstacle from a text-based ASCII art representation
 * 
 * @param mesh Reference to the mesh to operate on
 * @param asciiArt Vector of strings where each character represents a cell
 * @param x1 Physical x-coordinate of the ASCII art's left edge
 * @param y1 Physical y-coordinate of the ASCII art's bottom edge
 * @param x2 Physical x-coordinate of the ASCII art's right edge
 * @param y2 Physical y-coordinate of the ASCII art's top edge
 * @param obstacleChars String containing characters that represent obstacles
 * @param material Material to assign to the obstacle cells (nullptr for default)
 */
inline void createObstacleFromASCII(Mesh& mesh,
                             const std::vector<std::string>& asciiArt,
                             double x1, double y1, double x2, double y2,
                             const std::string& obstacleChars = "X#*",
                             std::shared_ptr<Material> material = nullptr) {
    
    // Check if ASCII art is empty
    if (asciiArt.empty() || asciiArt[0].empty()) {
        return;
    }
    
    // Create a binary matrix
    std::vector<std::vector<int>> matrix;
    matrix.reserve(asciiArt.size());
    
    for (const auto& line : asciiArt) {
        std::vector<int> row;
        row.reserve(line.length());
        
        for (char c : line) {
            // Check if character is in obstacleChars
            if (obstacleChars.find(c) != std::string::npos) {
                row.push_back(1); // Obstacle
            } else {
                row.push_back(0); // No obstacle
            }
        }
        
        matrix.push_back(row);
    }
    
    // Use the matrix function
    createObstacleFromMatrix(mesh, matrix, x1, y1, x2, y2, material);
}

/**
 * @struct PixmapImage
 * @brief Container for a parsed PPM/PGM image file
 */
struct PixmapImage {
    int width = 0;
    int height = 0;
    int maxValue = 0;
    bool isColor = false;
    std::vector<int> data;  // Flattened pixel data (for PGM: one value per pixel, for PPM: R,G,B triplets)
    
    // Get grayscale value for a pixel
    int getGrayscale(int x, int y) const {
        if (x < 0 || x >= width || y < 0 || y >= height) {
            return 0;
        }
        
        if (isColor) {
            // For color images, average the RGB values
            int idx = (y * width + x) * 3;
            return (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
        } else {
            // For grayscale images, return the value directly
            return data[y * width + x];
        }
    }
    
    // Check if a pixel is an obstacle based on threshold
    bool isObstacle(int x, int y, int threshold, bool invert) const {
        int value = getGrayscale(x, y);
        bool result = (value > threshold);
        return invert ? !result : result;
    }
};

/**
 * @brief Load a PPM or PGM image file
 * 
 * @param filename Path to the PPM/PGM file
 * @return Parsed image data
 * @throws std::runtime_error if there's an error reading the file
 */
inline PixmapImage loadPPMorPGM(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    
    PixmapImage image;
    std::string line;
    std::string magicNumber;
    
    // Skip comments and get magic number
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;  // Skip empty lines and comments
        }
        std::istringstream iss(line);
        iss >> magicNumber;
        break;
    }
    
    // Check magic number
    if (magicNumber == "P2") {
        // PGM (grayscale)
        image.isColor = false;
    } else if (magicNumber == "P3") {
        // PPM (color)
        image.isColor = true;
    } else {
        throw std::runtime_error("Unsupported file format. Expected P2 (PGM) or P3 (PPM)");
    }
    
    // Read width and height
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;  // Skip empty lines and comments
        }
        std::istringstream iss(line);
        iss >> image.width >> image.height;
        break;
    }
    
    // Read max value
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;  // Skip empty lines and comments
        }
        std::istringstream iss(line);
        iss >> image.maxValue;
        break;
    }
    
    // Calculate expected number of values
    int expectedValues = image.width * image.height;
    if (image.isColor) {
        expectedValues *= 3;  // RGB values for each pixel
    }
    
    // Read pixel data
    image.data.reserve(expectedValues);
    int value;
    
    while (file >> value) {
        image.data.push_back(value);
    }
    
    // Check if we read the expected number of values
    if (static_cast<int>(image.data.size()) != expectedValues) {
        throw std::runtime_error("Incorrect number of pixel values. Expected " + 
                                std::to_string(expectedValues) + ", got " + 
                                std::to_string(image.data.size()));
    }
    
    return image;
}

/**
 * @brief Create an obstacle from a PPM or PGM image file
 * 
 * @param mesh Reference to the mesh to operate on
 * @param filename Path to the PPM/PGM file
 * @param x1 Physical x-coordinate of the image's left edge
 * @param y1 Physical y-coordinate of the image's bottom edge
 * @param x2 Physical x-coordinate of the image's right edge
 * @param y2 Physical y-coordinate of the image's top edge
 * @param material Material to assign to the obstacle cells (nullptr for default)
 * @param threshold Brightness threshold (0 to maxValue) for obstacle detection
 * @param invertImage Whether to invert the obstacle definition (dark=obstacle vs light=obstacle)
 */
inline void createObstacleFromPPM(Mesh& mesh,
                           const std::string& filename,
                           double x1, double y1, double x2, double y2,
                           std::shared_ptr<Material> material = nullptr,
                           int threshold = -1,  // Default: half of maxValue
                           bool invertImage = false) {
    
    // Ensure coordinates are ordered correctly
    if (x1 > x2) std::swap(x1, x2);
    if (y1 > y2) std::swap(y1, y2);
    
    // Load the image
    PixmapImage image = loadPPMorPGM(filename);
    
    // If threshold wasn't specified, use half of maxValue
    if (threshold < 0) {
        threshold = image.maxValue / 2;
    }
    
    // Calculate physical size of each pixel
    double pixelWidth = (x2 - x1) / image.width;
    double pixelHeight = (y2 - y1) / image.height;
    
    // Process each pixel in the image
    for (int img_y = 0; img_y < image.height; img_y++) {
        // Flip y-coordinate so bottom of image maps to y1
        double physical_y = y1 + (image.height - img_y - 0.5) * pixelHeight;
        
        for (int img_x = 0; img_x < image.width; img_x++) {
            // Check if this pixel is an obstacle
            if (image.isObstacle(img_x, img_y, threshold, invertImage)) {
                // Calculate physical x-coordinate
                double physical_x = x1 + (img_x + 0.5) * pixelWidth;
                
                // Convert to grid indices
                size_t i = mesh.gridI(physical_x);
                size_t j = mesh.gridJ(physical_y);
                
                // Check if within bounds
                if (i < mesh.getNx() && j < mesh.getNy()) {
                    mesh.setCellAsObstacle(i, j, material);
                }
            }
        }
    }
}

/**
 * @brief Save a binary obstacle map as a PGM file
 * 
 * This is useful for visualizing or storing obstacles defined in the mesh.
 * 
 * @param mesh Reference to the mesh to read from
 * @param filename Path to save the PGM file
 * @param i1 Starting x-index in the grid
 * @param j1 Starting y-index in the grid
 * @param i2 Ending x-index in the grid (inclusive)
 * @param j2 Ending y-index in the grid (inclusive)
 * @return True if successful, false otherwise
 */
inline bool saveObstaclesToPGM(const Mesh& mesh,
                        const std::string& filename,
                        size_t i1 = 0, size_t j1 = 0,
                        size_t i2 = std::numeric_limits<size_t>::max(),
                        size_t j2 = std::numeric_limits<size_t>::max()) {
    
    // Limit to mesh bounds
    i1 = std::min(i1, mesh.getNx() - 1);
    j1 = std::min(j1, mesh.getNy() - 1);
    i2 = std::min(i2, mesh.getNx() - 1);
    j2 = std::min(j2, mesh.getNy() - 1);
    
    // Ensure correct order
    if (i1 > i2) std::swap(i1, i2);
    if (j1 > j2) std::swap(j1, j2);
    
    // Calculate dimensions
    size_t width = i2 - i1 + 1;
    size_t height = j2 - j1 + 1;
    
    // Open file for writing
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    // Write PGM header
    file << "P2\n";
    file << "# Generated from CFD mesh obstacle map\n";
    file << width << " " << height << "\n";
    file << "255\n";  // Max value
    
    // Write pixel data (inverted y-axis to match image convention)
    for (size_t j = j2; j >= j1; j--) {
        for (size_t i = i1; i <= i2; i++) {
            // Check if cell is an obstacle
            const Cell& cell = mesh.getCell(i, j);
            
            // Write value (255 for solid, 0 for fluid)
            if (cell.isObstacle()) {
                file << "255 ";
            } else {
                file << "0 ";
            }
        }
        file << "\n";
        
        // Handle case where j becomes negative (size_t underflow)
        if (j == 0) break;
    }
    
    return file.good();
}

/**
 * @brief Create a simple PGM file with an obstacle shape
 * 
 * This is useful for creating test obstacle files or templates.
 * 
 * @param filename Path to save the PGM file
 * @param width Width of the image
 * @param height Height of the image
 * @param shapeFunction Function that returns true if a point (x,y) is inside the shape
 * @param comment Optional comment to include in the PGM header
 * @return True if successful, false otherwise
 */
inline bool createPGMObstacleTemplate(
    const std::string& filename,
    int width, int height,
    std::function<bool(double,double)> shapeFunction,
    const std::string& comment = "CFD Obstacle Template") {
    
    // Open file for writing
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    // Write PGM header
    file << "P2\n";
    file << "# " << comment << "\n";
    file << width << " " << height << "\n";
    file << "255\n";  // Max value
    
    // Write pixel data
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Normalize coordinates to [0,1] range
            double nx = static_cast<double>(x) / (width - 1);
            double ny = static_cast<double>(y) / (height - 1);
            
            // Check if point is inside shape
            if (shapeFunction(nx, ny)) {
                file << "255 ";  // Inside shape (obstacle)
            } else {
                file << "0 ";    // Outside shape (fluid)
            }
        }
        file << "\n";
    }
    
    return file.good();
}

/**
 * @brief Create common obstacle template shapes as PGM files
 * 
 * @param baseName Base filename (without extension)
 * @param width Width of the image
 * @param height Height of the image
 * @return Vector of created filenames
 */
inline std::vector<std::string> createCommonObstacleTemplates(
    const std::string& baseName,
    int width = 512,
    int height = 512) {
    
    std::vector<std::string> filenames;
    
    // Circle obstacle
    std::string circleFile = baseName + "_circle.pgm";
    createPGMObstacleTemplate(circleFile, width, height,
        [](double x, double y) {
            // Center coordinates
            double cx = 0.5, cy = 0.5;
            double radius = 0.4;
            double dx = x - cx, dy = y - cy;
            return (dx*dx + dy*dy) < radius*radius;  // Inside circle
        },
        "Circle Obstacle Template");
    filenames.push_back(circleFile);
    
    // Ellipse obstacle
    std::string ellipseFile = baseName + "_ellipse.pgm";
    createPGMObstacleTemplate(ellipseFile, width, height,
        [](double x, double y) {
            // Center coordinates
            double cx = 0.5, cy = 0.5;
            double a = 0.4, b = 0.2;  // Semi-major and semi-minor axes
            double dx = x - cx, dy = y - cy;
            return (dx*dx/(a*a) + dy*dy/(b*b)) < 1.0;  // Inside ellipse
        },
        "Ellipse Obstacle Template");
    filenames.push_back(ellipseFile);
    
    // Rectangle obstacle
    std::string rectFile = baseName + "_rectangle.pgm";
    createPGMObstacleTemplate(rectFile, width, height,
        [](double x, double y) {
            return x >= 0.3 && x <= 0.7 && y >= 0.3 && y <= 0.7;  // Inside rectangle
        },
        "Rectangle Obstacle Template");
    filenames.push_back(rectFile);
    
    // Triangle obstacle
    std::string triangleFile = baseName + "_triangle.pgm";
    createPGMObstacleTemplate(triangleFile, width, height,
        [](double x, double y) {
            // Define the three vertices
            double x1 = 0.5, y1 = 0.8;  // Top
            double x2 = 0.2, y2 = 0.2;  // Bottom left
            double x3 = 0.8, y3 = 0.2;  // Bottom right
            
            // Barycentric coordinates
            double denominator = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3);
            double a = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denominator;
            double b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denominator;
            double c = 1 - a - b;
            
            return (a >= 0) && (b >= 0) && (c >= 0);  // Inside triangle
        },
        "Triangle Obstacle Template");
    filenames.push_back(triangleFile);
    
    // NACA airfoil
    std::string airfoilFile = baseName + "_airfoil.pgm";
    createPGMObstacleTemplate(airfoilFile, width, height,
        [](double x, double y) {
            // Center and normalize
            double nx = x * 2.0 - 0.5;  // [0,1] -> [-0.5,1.5] to show full airfoil
            double ny = (y - 0.5) * 0.5; // Scale y for better visualization
            
            // NACA 0012 airfoil
            if (nx < 0.0 || nx > 1.0) return false;  // Outside airfoil length
            
            double t = 0.12;  // 12% thickness
            double yt = t/0.2 * (0.2969*sqrt(nx) - 0.1260*nx - 0.3516*nx*nx + 0.2843*pow(nx,3) - 0.1015*pow(nx,4));
            
            return std::abs(ny) <= yt;  // Inside airfoil thickness
        },
        "NACA Airfoil Obstacle Template");
    filenames.push_back(airfoilFile);
    
    return filenames;
}


} // namespace MeshUtils
  
