/**
 * ====================================================================
 * MeshVisualizer - Module for visualizing CFD mesh data
 * ====================================================================
 */
#pragma once

#include "Mesh.h"
#include <string>
#include <vector>
#include <map>

/**
 * @namespace MeshVisualizer
 * @brief Provides functions for visualizing mesh data in various formats
 */
namespace MeshVisualizer {

    /**
     * @enum ColorScheme
     * @brief Different color schemes for visualization
     */
    enum class ColorScheme {
        RAINBOW,       ///< Rainbow color scale (blue to red)
        BLUE_RED,      ///< Blue to red color scale
        GRAYSCALE,     ///< Grayscale color scale
        PLASMA,        ///< Plasma color scale (purple to yellow)
        VIRIDIS,       ///< Viridis color scale (blue to yellow)
        TERRAIN        ///< Terrain color scale (blue, green, brown)
    };

    /**
     * @brief Generate SVG representation of the mesh
     * @param mesh Reference to the mesh
     * @param showVelocity Include velocity vectors in the visualization
     * @param showPressure Include pressure values in the visualization
     * @param showTemperature Include temperature values in the visualization
     * @param showBoundaries Highlight boundary conditions
     * @param scale Scale factor for the visualization
     * @return String containing SVG XML
     */
    std::string toSVG(
        const Mesh& mesh,
        bool showVelocity = true, 
        bool showPressure = true,
        bool showTemperature = true, 
        bool showBoundaries = true,
        double scale = 10.0);
    
    /**
     * @brief Generate a heatmap SVG of a specific field
     * @param mesh Reference to the mesh
     * @param field Vector of field values
     * @param fieldName Name of the field for labeling
     * @param minValue Minimum value for color scale (or NAN for auto)
     * @param maxValue Maximum value for color scale (or NAN for auto)
     * @param colorScheme Color scheme to use
     * @param scale Scale factor for the visualization
     * @return String containing SVG XML
     */
    std::string fieldToSVG(
        const Mesh& mesh,
        const std::vector<double>& field,
        const std::string& fieldName,
        double minValue = NAN, 
        double maxValue = NAN,
        ColorScheme colorScheme = ColorScheme::RAINBOW,
        double scale = 10.0);
    
    /**
     * @brief Generate PNG images for all basic fields of the mesh
     * @param mesh Reference to the mesh
     * @param baseFilename Base filename for the output images (field name will be appended)
     * @param width Image width in pixels
     * @param height Image height in pixels (0 for auto-calculated based on mesh aspect ratio)
     * @param colorScheme Color scheme to use for continuous fields
     * @return Number of images created
     */
    int exportAllFieldsToPNG(
        const Mesh& mesh,
        const std::string& baseFilename,
        int width = 800,
        int height = 0,
        ColorScheme colorScheme = ColorScheme::RAINBOW);
    
    /**
     * @brief Generate a PNG image of a specific field
     * @param mesh Reference to the mesh
     * @param field Vector of field values
     * @param filename Output filename
     * @param fieldName Name of the field for labeling
     * @param width Image width in pixels
     * @param height Image height in pixels (0 for auto-calculated based on mesh aspect ratio)
     * @param minValue Minimum value for color scale (or NAN for auto)
     * @param maxValue Maximum value for color scale (or NAN for auto)
     * @param colorScheme Color scheme to use
     * @param showGrid Whether to draw grid lines
     * @return True if successful
     */
    bool fieldToPNG(
        const Mesh& mesh,
        const std::vector<double>& field,
        const std::string& filename,
        const std::string& fieldName,
        int width = 800,
        int height = 0,
        double minValue = NAN, 
        double maxValue = NAN,
        ColorScheme colorScheme = ColorScheme::RAINBOW,
        bool showGrid = true);
    
    /**
     * @brief Generate a PNG image of a vector field
     * @param mesh Reference to the mesh
     * @param vectorFieldX X-component of the vector field
     * @param vectorFieldY Y-component of the vector field
     * @param filename Output filename
     * @param fieldName Name of the field for labeling
     * @param width Image width in pixels
     * @param height Image height in pixels (0 for auto-calculated based on mesh aspect ratio)
     * @param decimation Decimation factor for arrow density (1 = every cell)
     * @param colorScheme Color scheme to use for vector magnitude
     * @param showGrid Whether to draw grid lines
     * @return True if successful
     */
    bool vectorFieldToPNG(
        const Mesh& mesh,
        const std::vector<double>& vectorFieldX,
        const std::vector<double>& vectorFieldY,
        const std::string& filename,
        const std::string& fieldName,
        int width = 800,
        int height = 0,
        int decimation = 3,
        ColorScheme colorScheme = ColorScheme::RAINBOW,
        bool showGrid = true);
    
    /**
     * @brief Create a PNG visualization of the mesh setup for validation
     * 
     * This creates an image where different cell types are clearly visible:
     * - Different boundary types have different colors
     * - Obstacles are shown distinctly
     * - Cell IDs can be overlaid
     * - Materials can be distinguished
     * 
     * Intended for validating the problem setup visually.
     * 
     * @param mesh Reference to the mesh
     * @param filename Output filename
     * @param width Image width in pixels
     * @param height Image height in pixels (0 for auto-calculated based on mesh aspect ratio)
     * @param showCellIndices Whether to overlay cell indices
     * @param showMaterialTypes Whether to use different colors for materials
     * @param showBoundaryTypes Whether to distinguish different boundary types
     * @param showBoundaryNames Whether to overlay boundary condition names
     * @return True if successful
     */
    bool meshSetupToPNG(
        const Mesh& mesh,
        const std::string& filename,
        int width = 800,
        int height = 0,
        bool showCellIndices = false,
        bool showMaterialTypes = true,
        bool showBoundaryTypes = true,
        bool showBoundaryNames = false);
    
    /**
     * @brief Generate VTK file for ParaView or other visualization tools
     * @param mesh Reference to the mesh
     * @param filename Output VTK filename
     * @param includedFields Vector of field names to include (empty = all basic fields)
     * @return True if the file was written successfully
     */
    bool toVTK(
        const Mesh& mesh,
        const std::string& filename,
        const std::vector<std::string>& includedFields = {});
    
    /**
     * @brief Helper function to map a value to a color using a colormap
     * @param value The value to map
     * @param minValue Minimum value in the range
     * @param maxValue Maximum value in the range
     * @param colorScheme Color scheme to use
     * @return RGB color values as array [r, g, b] with values 0-255
     */
    std::array<unsigned char, 3> valueToColor(
        double value,
        double minValue,
        double maxValue,
        ColorScheme colorScheme = ColorScheme::RAINBOW);
}
