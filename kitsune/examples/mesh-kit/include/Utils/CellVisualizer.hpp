/**
 * @file CellVisualizer.hpp
 * @brief Implementation of the cell visualization utility
 */

#include <iomanip>
#include <fstream>
#include <cmath>

namespace CellVisualization {

/**
 * @brief Create an SVG circle element
 * 
 * @param cx Center x-coordinate
 * @param cy Center y-coordinate
 * @param r Radius
 * @param fill Fill color
 * @param stroke Stroke color
 * @param strokeWidth Stroke width
 * @return std::string SVG circle element
 */
inline std::string createCircle(double cx, double cy, double r, 
                             const std::string& fill, 
                             const std::string& stroke = "#000000", 
                             double strokeWidth = 1.0) {
    std::stringstream ss;
    ss << "<circle cx=\"" << cx << "\" cy=\"" << cy << "\" r=\"" << r 
       << "\" fill=\"" << fill << "\" stroke=\"" << stroke 
       << "\" stroke-width=\"" << strokeWidth << "\" />";
    return ss.str();
}

/**
 * @brief Create an SVG rectangle element
 * 
 * @param x X-coordinate
 * @param y Y-coordinate
 * @param width Width
 * @param height Height
 * @param fill Fill color
 * @param stroke Stroke color
 * @param strokeWidth Stroke width
 * @return std::string SVG rectangle element
 */
inline std::string createRect(double x, double y, double width, double height, 
                           const std::string& fill, 
                           const std::string& stroke = "#000000", 
                           double strokeWidth = 1.0) {
    std::stringstream ss;
    ss << "<rect x=\"" << x << "\" y=\"" << y << "\" width=\"" << width 
       << "\" height=\"" << height << "\" fill=\"" << fill 
       << "\" stroke=\"" << stroke << "\" stroke-width=\"" << strokeWidth << "\" />";
    return ss.str();
}

/**
 * @brief Create an SVG line element
 * 
 * @param x1 Start x-coordinate
 * @param y1 Start y-coordinate
 * @param x2 End x-coordinate
 * @param y2 End y-coordinate
 * @param stroke Stroke color
 * @param strokeWidth Stroke width
 * @param dashArray Optional dash array (e.g., "5,5" for dashed line)
 * @return std::string SVG line element
 */
inline std::string createLine(double x1, double y1, double x2, double y2, 
                           const std::string& stroke = "#000000", 
                           double strokeWidth = 1.0,
                           const std::string& dashArray = "") {
    std::stringstream ss;
    ss << "<line x1=\"" << x1 << "\" y1=\"" << y1 << "\" x2=\"" << x2 
       << "\" y2=\"" << y2 << "\" stroke=\"" << stroke 
       << "\" stroke-width=\"" << strokeWidth << "\"";
    
    if (!dashArray.empty()) {
        ss << " stroke-dasharray=\"" << dashArray << "\"";
    }
    
    ss << " />";
    return ss.str();
}

/**
 * @brief Create an SVG text element
 * 
 * @param x X-coordinate
 * @param y Y-coordinate
 * @param text Text content
 * @param fontSize Font size
 * @param fill Text color
 * @param anchor Text anchor (start, middle, end)
 * @return std::string SVG text element
 */
inline std::string createText(double x, double y, const std::string& text, 
                           double fontSize = 12, 
                           const std::string& fill = "#000000",
                           const std::string& anchor = "middle") {
    std::stringstream ss;
    ss << "<text x=\"" << x << "\" y=\"" << y << "\" font-size=\"" << fontSize 
       << "\" fill=\"" << fill << "\" text-anchor=\"" << anchor 
       << "\" dominant-baseline=\"middle\">" << text << "</text>";
    return ss.str();
}

/**
 * @brief Format a double value for display
 * 
 * @param value Value to format
 * @param precision Number of decimal places
 * @return std::string Formatted value
 */
inline std::string formatDouble(double value, int precision = 2) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(precision) << value;
    return ss.str();
}

std::string generateCellSVG(const CellBase* cell, const CellVisualizationConfig& config) {
    if (!cell || !cell->mesh()) {
        return "<svg width=\"100\" height=\"100\"><text x=\"50\" y=\"50\" text-anchor=\"middle\">Invalid Cell</text></svg>";
    }
    
    const int cellSize = config.cellSize;
    const int margin = config.margin;
    const int totalSize = cellSize + 2 * margin;
    
    std::stringstream svg;
    svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << totalSize 
        << "\" height=\"" << totalSize << "\" viewBox=\"0 0 " << totalSize 
        << " " << totalSize << "\">";
    
    // Background
    svg << createRect(0, 0, totalSize, totalSize, config.backgroundColor);
    
    // Cell rectangle
    svg << createRect(margin, margin, cellSize, cellSize, config.cellColor, config.borderColor, 2.0);
    
    // Get cell information
    int i = cell->i();
    int j = cell->j();
    auto [physX, physY] = cell->position();
    double dx = cell->mesh()->dx();
    double dy = cell->mesh()->dy();
    
    // Center point
    int centerX = margin + cellSize / 2;
    int centerY = margin + cellSize / 2;
    svg << createCircle(centerX, centerY, 4, config.centerMarkerColor);
    
    // Calculate vertex positions
    int vertexSize = 3;
    std::vector<std::pair<int, int>> vertices = {
        {margin, margin},                     // SW
        {margin + cellSize, margin},          // SE
        {margin + cellSize, margin + cellSize}, // NE
        {margin, margin + cellSize}           // NW
    };
    
    // Draw vertices
    for (const auto& [vx, vy] : vertices) {
        svg << createCircle(vx, vy, vertexSize, config.vertexMarkerColor);
    }
    
    // Draw edge midpoints
    std::vector<std::pair<int, int>> edges = {
        {margin + cellSize / 2, margin},              // South
        {margin + cellSize, margin + cellSize / 2},    // East
        {margin + cellSize / 2, margin + cellSize},    // North
        {margin, margin + cellSize / 2}                // West
    };
    
    for (const auto& [ex, ey] : edges) {
        svg << createCircle(ex, ey, vertexSize, config.edgeMarkerColor);
    }
    
    // Add labels for cell indices if enabled
    if (config.showCellIndices) {
        std::string indicesLabel = "Cell (" + std::to_string(i) + ", " + std::to_string(j) + ")";
        svg << createText(centerX, margin - 15, indicesLabel, config.fontSize, config.textColor);
    }
    
    // Add physical dimensions if enabled
    if (config.showPhysicalDimensions) {
        std::string dimLabel = "Physical size: " + formatDouble(dx) + " x " + formatDouble(dy);
        svg << createText(centerX, margin + cellSize + 15, dimLabel, config.fontSize, config.textColor);
    }
    
    // Add physical coordinates if enabled
    if (config.showCoordinates) {
        std::string coordLabel = "Position: (" + formatDouble(physX) + ", " + formatDouble(physY) + ")";
        svg << createText(centerX, margin + cellSize + 30, coordLabel, config.fontSize, config.textColor);
    }
    
    // Add field labels based on provided callbacks
    
    // Center field
    if (config.centerFieldLabel) {
        std::string label = config.centerFieldLabel(i, j);
        if (!label.empty()) {
            svg << createText(centerX, centerY, label, config.fontSize, config.textColor);
        }
    }
    
    // Edge fields
    if (config.southEdgeFieldLabel) {
        std::string label = config.southEdgeFieldLabel(i, j);
        if (!label.empty()) {
            svg << createText(edges[0].first, edges[0].second - 10, label, config.fontSize, config.textColor);
        }
    }
    
    if (config.eastEdgeFieldLabel) {
        std::string label = config.eastEdgeFieldLabel(i, j);
        if (!label.empty()) {
            svg << createText(edges[1].first + 10, edges[1].second, label, config.fontSize, config.textColor);
        }
    }
    
    if (config.northEdgeFieldLabel) {
        std::string label = config.northEdgeFieldLabel(i, j);
        if (!label.empty()) {
            svg << createText(edges[2].first, edges[2].second + 10, label, config.fontSize, config.textColor);
        }
    }
    
    if (config.westEdgeFieldLabel) {
        std::string label = config.westEdgeFieldLabel(i, j);
        if (!label.empty()) {
            svg << createText(edges[3].first - 10, edges[3].second, label, config.fontSize, config.textColor);
        }
    }
    
    // Vertex fields
    if (config.swVertexFieldLabel) {
        std::string label = config.swVertexFieldLabel(i, j);
        if (!label.empty()) {
            svg << createText(vertices[0].first - 10, vertices[0].second - 10, label, config.fontSize, config.textColor);
        }
    }
    
    if (config.seVertexFieldLabel) {
        std::string label = config.seVertexFieldLabel(i, j);
        if (!label.empty()) {
            svg << createText(vertices[1].first + 10, vertices[1].second - 10, label, config.fontSize, config.textColor);
        }
    }
    
    if (config.neVertexFieldLabel) {
        std::string label = config.neVertexFieldLabel(i, j);
        if (!label.empty()) {
            svg << createText(vertices[2].first + 10, vertices[2].second + 10, label, config.fontSize, config.textColor);
        }
    }
    
    if (config.nwVertexFieldLabel) {
        std::string label = config.nwVertexFieldLabel(i, j);
        if (!label.empty()) {
            svg << createText(vertices[3].first - 10, vertices[3].second + 10, label, config.fontSize, config.textColor);
        }
    }
    
    // Add legend
    svg << createCircle(margin + 10, totalSize - margin + 10, 4, config.centerMarkerColor);
    svg << createText(margin + 25, totalSize - margin + 10, "Cell Center", config.fontSize, config.textColor, "start");
    
    svg << createCircle(margin + 100, totalSize - margin + 10, vertexSize, config.vertexMarkerColor);
    svg << createText(margin + 115, totalSize - margin + 10, "Vertex", config.fontSize, config.textColor, "start");
    
    svg << createCircle(margin + 180, totalSize - margin + 10, vertexSize, config.edgeMarkerColor);
    svg << createText(margin + 195, totalSize - margin + 10, "Edge", config.fontSize, config.textColor, "start");
    
    svg << "</svg>";
    return svg.str();
}

std::string generateRegionSVG(const CellBase* centralCell, int neighborhoodSize, 
                            const CellVisualizationConfig& config) {
    if (!centralCell || !centralCell->mesh()) {
        return "<svg width=\"100\" height=\"100\"><text x=\"50\" y=\"50\" text-anchor=\"middle\">Invalid Cell</text></svg>";
    }
    
    const int cellSize = config.cellSize;
    const int margin = config.margin;
    
    // Calculate total grid size
    int gridSize = 2 * neighborhoodSize + 1;
    int totalWidth = gridSize * cellSize + 2 * margin;
    int totalHeight = gridSize * cellSize + 2 * margin;
    
    // Central cell indices
    int centralI = centralCell->i();
    int centralJ = centralCell->j();
    
    // Get mesh info
    MeshBase* mesh = centralCell->mesh();
    double dx = mesh->dx();
    double dy = mesh->dy();
    
    std::stringstream svg;
    svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << totalWidth 
        << "\" height=\"" << totalHeight << "\" viewBox=\"0 0 " << totalWidth 
        << " " << totalHeight << "\">";
    
    // Background
    svg << createRect(0, 0, totalWidth, totalHeight, config.backgroundColor);
    
    // Calculate the min and max indices to display
    int minI = centralI - neighborhoodSize;
    int maxI = centralI + neighborhoodSize;
    int minJ = centralJ - neighborhoodSize;
    int maxJ = centralJ + neighborhoodSize;
    
    // Draw grid cells
    for (int j = minJ; j <= maxJ; ++j) {
        for (int i = minI; i <= maxI; ++i) {
            // Check if this cell exists
            CellBase* cell = mesh->getCell(i, j);
            if (!cell) continue;
            
            // Calculate cell position in SVG
            int cellX = margin + (i - minI) * cellSize;
            int cellY = margin + (maxJ - j) * cellSize; // Flip Y to match standard orientation
            
            // Draw cell rectangle
            std::string fillColor = (i == centralI && j == centralJ) 
                                    ? "#ffeeee"  // Highlight central cell
                                    : config.cellColor;
            svg << createRect(cellX, cellY, cellSize, cellSize, fillColor, config.borderColor, 1.0);
            
            // Cell center
            int centerX = cellX + cellSize / 2;
            int centerY = cellY + cellSize / 2;
            svg << createCircle(centerX, centerY, 3, config.centerMarkerColor);
            
            // Cell indices
            svg << createText(centerX, centerY - 15, "(" + std::to_string(i) + "," + std::to_string(j) + ")", 
                           config.fontSize - 2, config.textColor);
            
            // Add field labels based on provided callbacks
            if (config.centerFieldLabel) {
                std::string label = config.centerFieldLabel(i, j);
                if (!label.empty()) {
                    svg << createText(centerX, centerY + 15, label, config.fontSize - 2, config.textColor);
                }
            }
            
            // Add edge markers and labels
            std::vector<std::tuple<int, int, std::function<std::string(int,int)>>> edges = {
                {centerX, cellY + cellSize, config.northEdgeFieldLabel},  // North
                {cellX + cellSize, centerY, config.eastEdgeFieldLabel},   // East
                {centerX, cellY, config.southEdgeFieldLabel},             // South
                {cellX, centerY, config.westEdgeFieldLabel}               // West
            };
            
            for (const auto& [ex, ey, labelFunc] : edges) {
                svg << createCircle(ex, ey, 2, config.edgeMarkerColor);
                if (labelFunc) {
                    std::string label = labelFunc(i, j);
                    if (!label.empty()) {
                        // Position the label based on which edge it is
                        if (ex == centerX && ey == cellY) {  // South edge
                            svg << createText(ex, ey - 10, label, config.fontSize - 2, config.textColor);
                        } else if (ex == centerX && ey == cellY + cellSize) {  // North edge
                            svg << createText(ex, ey + 10, label, config.fontSize - 2, config.textColor);
                        } else if (ex == cellX && ey == centerY) {  // West edge
                            svg << createText(ex - 10, ey, label, config.fontSize - 2, config.textColor);
                        } else {  // East edge
                            svg << createText(ex + 10, ey, label, config.fontSize - 2, config.textColor);
                        }
                    }
                }
            }
        }
    }
    
    // Add legend
    svg << createCircle(margin + 10, totalHeight - margin + 10, 3, config.centerMarkerColor);
    svg << createText(margin + 25, totalHeight - margin + 10, "Cell Center", config.fontSize, config.textColor, "start");
    
    svg << createCircle(margin + 120, totalHeight - margin + 10, 2, config.edgeMarkerColor);
    svg << createText(margin + 135, totalHeight - margin + 10, "Edge", config.fontSize, config.textColor, "start");
    
    // Add physical scale information
    std::string scaleInfo = "Physical dimensions: " + formatDouble(dx) + " x " + formatDouble(dy) + " per cell";
    svg << createText(totalWidth / 2, totalHeight - margin / 2, scaleInfo, config.fontSize, config.textColor);
    
    svg << "</svg>";
    return svg.str();
}

template<typename MeshType, typename T>
std::function<std::string(int i, int j)> createCenterFieldLabelProvider(
    const MeshType* mesh,
    const std::function<const T&(const MeshType*, int, int)>& fieldAccessor,
    const std::string& fieldName) {
    
    return [mesh, fieldAccessor, fieldName](int i, int j) -> std::string {
        if (!mesh) return "";
        
        try {
            const T& value = fieldAccessor(mesh, i, j);
            std::stringstream ss;
            if (!fieldName.empty()) {
                ss << fieldName << ": ";
            }
            ss << formatDouble(static_cast<double>(value));
            return ss.str();
        } catch (const std::exception&) {
            return "";
        }
    };
}

template<typename MeshType, typename T>
std::function<std::string(int i, int j)> createVertexFieldLabelProvider(
    const MeshType* mesh,
    const std::function<const T&(const MeshType*, int, int)>& fieldAccessor,
    const std::string& fieldName) {
    
    return [mesh, fieldAccessor, fieldName](int i, int j) -> std::string {
        if (!mesh) return "";
        
        try {
            const T& value = fieldAccessor(mesh, i, j);
            std::stringstream ss;
            if (!fieldName.empty()) {
                ss << fieldName << ": ";
            }
            ss << formatDouble(static_cast<double>(value));
            return ss.str();
        } catch (const std::exception&) {
            return "";
        }
    };
}

template<typename MeshType, typename T>
std::function<std::string(int i, int j)> createEdgeFieldLabelProvider(
    const MeshType* mesh,
    const std::function<const T&(const MeshType*, int, int)>& fieldAccessor,
    const std::string& fieldName) {
    
    return [mesh, fieldAccessor, fieldName](int i, int j) -> std::string {
        if (!mesh) return "";
        
        try {
            const T& value = fieldAccessor(mesh, i, j);
            std::stringstream ss;
            if (!fieldName.empty()) {
                ss << fieldName << ": ";
            }
            ss << formatDouble(static_cast<double>(value));
            return ss.str();
        } catch (const std::exception&) {
            return "";
        }
    };
}

bool saveSVG(const std::string& filename, const std::string& svg) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    file << svg;
    return !file.bad();
}

std::string createHTMLDocument(const std::string& svg, const std::string& title) {
    std::stringstream html;
    html << "<!DOCTYPE html>\n";
    html << "<html lang=\"en\">\n";
    html << "<head>\n";
    html << "    <meta charset=\"UTF-8\">\n";
    html << "    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n";
    html << "    <title>" << title << "</title>\n";
    html << "    <style>\n";
    html << "        body { font-family: Arial, sans-serif; margin: 20px; }\n";
    html << "        h1 { color: #333; }\n";
    html << "        .svg-container { margin: 20px 0; }\n";
    html << "    </style>\n";
    html << "</head>\n";
    html << "<body>\n";
    html << "    <h1>" << title << "</h1>\n";
    html << "    <div class=\"svg-container\">\n";
    html << "        " << svg << "\n";
    html << "    </div>\n";
    html << "</body>\n";
    html << "</html>";
    
    return html.str();
}

} // namespace CellVisualization


