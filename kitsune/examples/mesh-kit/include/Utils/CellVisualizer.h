/**
 * @file CellVisualizer.h
 * @brief Utility functions for visualizing cell structure and fields
 * 
 * This file provides functions to generate SVG representations of cells,
 * including field placement, physical dimensions, and other debugging information.
 */

#ifndef CELL_VISUALIZER_H
#define CELL_VISUALIZER_H

#include "CellBase.h"
#include <string>
#include <sstream>
#include <map>
#include <vector>
#include <functional>

namespace CellVisualization {

/**
 * @brief Configuration for cell visualization
 */
struct CellVisualizationConfig {
    int cellSize = 200;              ///< Size of cell in pixels
    int margin = 40;                 ///< Margin around cell in pixels
    std::string backgroundColor = "#f0f0f0"; ///< Background color
    std::string cellColor = "#ffffff";       ///< Cell color
    std::string borderColor = "#000000";     ///< Border color
    std::string textColor = "#000000";       ///< Text color
    std::string centerMarkerColor = "#ff0000"; ///< Color for center marker
    std::string vertexMarkerColor = "#0000ff"; ///< Color for vertex markers
    std::string edgeMarkerColor = "#00aa00";   ///< Color for edge markers
    int fontSize = 12;               ///< Font size for labels
    bool showGrid = true;            ///< Whether to show grid lines
    bool showCoordinates = true;     ///< Whether to show coordinates
    bool showPhysicalDimensions = true; ///< Whether to show physical dimensions
    bool showCellIndices = true;     ///< Whether to show cell indices
    
    // Field decoration callbacks - these functions return field information
    std::function<std::string(int i, int j)> centerFieldLabel; ///< Center field label provider
    std::function<std::string(int i, int j)> northEdgeFieldLabel; ///< North edge field label provider
    std::function<std::string(int i, int j)> eastEdgeFieldLabel;  ///< East edge field label provider
    std::function<std::string(int i, int j)> southEdgeFieldLabel; ///< South edge field label provider
    std::function<std::string(int i, int j)> westEdgeFieldLabel;  ///< West edge field label provider
    std::function<std::string(int i, int j)> neVertexFieldLabel;  ///< NE vertex field label provider
    std::function<std::string(int i, int j)> nwVertexFieldLabel;  ///< NW vertex field label provider
    std::function<std::string(int i, int j)> seVertexFieldLabel;  ///< SE vertex field label provider
    std::function<std::string(int i, int j)> swVertexFieldLabel;  ///< SW vertex field label provider
};

/**
 * @brief Generate an SVG representation of a cell
 * 
 * @param cell The cell to visualize
 * @param config Visualization configuration
 * @return std::string SVG representation of the cell
 */
std::string generateCellSVG(const CellBase* cell, const CellVisualizationConfig& config = {});

/**
 * @brief Generate an SVG representation of a multi-cell region
 * 
 * @param centralCell Central cell of the region to visualize
 * @param neighborhoodSize Number of cells to include in each direction
 * @param config Visualization configuration
 * @return std::string SVG representation of the cell region
 */
std::string generateRegionSVG(const CellBase* centralCell, int neighborhoodSize = 1, 
                            const CellVisualizationConfig& config = {});

/**
 * @brief Field label provider for cell-centered fields
 * 
 * Creates a function that can be assigned to config.centerFieldLabel
 * 
 * @tparam MeshType Type of the mesh
 * @tparam T Field data type
 * @param mesh Mesh containing the field
 * @param fieldAccessor Function to access the field from the mesh
 * @param fieldName Optional name to display with the field value
 * @return std::function Field label provider function
 */
template<typename MeshType, typename T>
std::function<std::string(int i, int j)> createCenterFieldLabelProvider(
    const MeshType* mesh,
    const std::function<const T&(const MeshType*, int, int)>& fieldAccessor,
    const std::string& fieldName = "");

/**
 * @brief Field label provider for vertex-centered fields
 * 
 * Creates a function that can be assigned to vertex field label providers
 * 
 * @tparam MeshType Type of the mesh
 * @tparam T Field data type
 * @param mesh Mesh containing the field
 * @param fieldAccessor Function to access the field from the mesh
 * @param fieldName Optional name to display with the field value
 * @return std::function Field label provider function
 */
template<typename MeshType, typename T>
std::function<std::string(int i, int j)> createVertexFieldLabelProvider(
    const MeshType* mesh,
    const std::function<const T&(const MeshType*, int, int)>& fieldAccessor,
    const std::string& fieldName = "");

/**
 * @brief Field label provider for edge-centered fields
 * 
 * Creates a function that can be assigned to edge field label providers
 * 
 * @tparam MeshType Type of the mesh
 * @tparam T Field data type
 * @param mesh Mesh containing the field
 * @param fieldAccessor Function to access the field from the mesh
 * @param fieldName Optional name to display with the field value
 * @return std::function Field label provider function
 */
template<typename MeshType, typename T>
std::function<std::string(int i, int j)> createEdgeFieldLabelProvider(
    const MeshType* mesh,
    const std::function<const T&(const MeshType*, int, int)>& fieldAccessor,
    const std::string& fieldName = "");

/**
 * @brief Save SVG to a file
 * 
 * @param filename Filename to save to
 * @param svg SVG content to save
 * @return bool True if successful
 */
bool saveSVG(const std::string& filename, const std::string& svg);

/**
 * @brief Create a complete HTML document containing the SVG
 * 
 * @param svg SVG content to include
 * @param title Optional title for the document
 * @return std::string HTML document
 */
std::string createHTMLDocument(const std::string& svg, const std::string& title = "Cell Visualization");

} // namespace CellVisualization

#include "CellVisualizer.hpp"  // Template implementation

#endif // CELL_VISUALIZER_H


