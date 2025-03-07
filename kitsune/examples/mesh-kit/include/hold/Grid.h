/**
 * ====================================================================
 * Grid Class - CFD Implementation Documentation
 * ====================================================================
 *
 * The Grid class manages the spatial discretization of the computational
 * domain, orchestrating collections of Cell objects and providing methods for
 * domain setup, boundary condition application, and field operation handling.
 *
 * Relationship with Cell and Material Classes:
 * ------------------------------------------
 * - Cell Management:
 *   Grid serves as a container for an array of Cell objects, arranged in a 2D
 * structured mesh. It provides efficient indexed access to cells (via getCell)
 * and direct array access for high-performance operations (via getCellData).
 * Cell properties are accessed and modified through the Grid's coordinate
 * system.
 *
 * - Material Integration:
 *   While Grid doesn't store Materials directly, it coordinates Material
 * assignment to Cells through methods like setCircularObstacle and
 * setMaterialRegion. It enforces that each cell has an associated Material to
 * define its physical behavior.
 *
 * Primary Use Cases:
 * ----------------
 * 1. Simulation Domain Management
 *    - Establishes the mapping between physical space and computational grid
 *    - Handles boundary detection and cell type classification
 *    - Provides coordinate transformations between grid indices and physical
 *      coordinates
 *    - Supports unit conversions for all domain dimensions
 *
 * 2. Flow Field Initialization and Manipulation
 *    - Applies complex flow patterns (uniform, shear, parabolic, vortex, and
 *      jet flows)
 *    - Sets up temperature, pressure, and density distributions
 *    - Establishes gradients and stratifications for realistic initial
 *      conditions
 *    - Enables interpolation for sub-grid accuracy
 *
 * 3. Geometry and Obstacle Definition
 *    - Creates boundary cells around the domain perimeter
 *    - Supports circular obstacle definition with appropriate material
 *      properties
 *    - Enforces no-slip conditions at solid boundaries
 *    - Provides foundation for more complex geometry specifications
 *
 * 4. Field Access and Analysis
 *    - Retrieves complete field data for visualization and post-processing
 *    - Calculates derived quantities such as flow divergence
 *    - Provides consistent access to cell properties across the domain
 *    - Supports unit conversion for all physical quantities
 *
 * Structure and Organization:
 * -------------------------
 *    +---------------+           +---------------+
 *    | Grid          |           | Cell          |
 *    |---------------|           |---------------|
 *    | m_nx, m_ny    |           | m_type        |
 *    | m_width       |           | m_temperature |
 *    | m_height      | contains  | m_pressure    |
 *    | m_origin_x/y  |---------->| m_velocity    |
 *    | m_dx, m_dy    |           | m_material    |
 *    | m_cells       |           |               |
 *    +---------------+           +---------------+
 *                                       |
 *                                       | references
 *                                       v
 *                                +---------------+
 *                                | Material      |
 *                                |---------------|
 *                                | m_type        |
 *                                | m_properties  |
 *                                | m_models      |
 *                                +---------------+
 *
 * Performance Considerations:
 * -------------------------
 * - Uses 1D vector for storage with index calculation for optimal memory access
 * - Provides direct array access for high-performance solver implementations
 * - Uses inline methods for frequently called coordinate transformations
 * - Implements bounds checking only in debug builds for production performance
 *
 * Implementation Notes:
 * ------------------
 * - Grid cells are stored in row-major order (x varies fastest) for efficient
 *   access
 * - Boundary cells are automatically marked during grid creation
 * - Uses SI units internally (meters, Kelvin, Pascal, kg/m³)
 * - Provides comprehensive unit conversion methods with the Units helper class
 *
 * LIMITATIONS:
 * -----------
 * 1. CURRENT IMPLEMENTATION:
 *    - Supports only 2D structured rectangular grids with uniform spacing
 *    - Limited to Cartesian coordinates (no curvilinear or body-fitted meshes)
 *    - No built-in support for adaptive mesh refinement or multi-resolution
 *    - Circular obstacles are the only built-in geometric primitive
 *
 * 2. BOUNDARY CONDITION HANDLING:
 *    - Basic boundary condition types (wall, inlet, outlet) through cell flags
 *    - No specialized boundary condition treatment for complex physics
 *    - Limited support for moving boundaries or deforming domains
 *    - No built-in periodic boundary conditions
 *
 * 3. PARALLELIZATION:
 *    - No built-in domain decomposition for parallel computing
 *    - No ghost/halo cell management for parallel communication
 *    - Single-process implementation limits scalability to large problems
 *
 * POSSIBLE FUTURE EXTENSIONS:
 * -------------------------
 * 1. ADVANCED MESH CAPABILITIES:
 *    - Non-uniform grid spacing for focused resolution
 *    - Block-structured mesh with hanging nodes
 *    - Curvilinear coordinates for complex geometries
 *    - Adaptive mesh refinement for dynamic flow features
 *
 * 2. ENHANCED BOUNDARY CONDITIONS:
 *    - Specialized inlet profiles (developed flow, turbulent)
 *    - Advanced wall functions for turbulence modeling
 *    - Radiation and conjugate heat transfer boundaries
 *    - Wave-transparent (non-reflecting) outlet conditions
 *
 * 3. PERFORMANCE OPTIMIZATIONS:
 *    - Multi-threading support for shared-memory parallelism
 *    - MPI integration for distributed computing
 *    - GPU acceleration for compute-intensive operations
 *    - Cache-aware algorithms for improved memory performance
 *
 * 4. PHYSICAL MODELS:
 *    - Multi-phase flow capabilities
 *    - Free surface tracking methods
 *    - Turbulence model integration
 *    - Particle tracking for dispersed phases
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

// Include updated Cell and Material classes
#include "Cell.h"
#include "Material.h"

/**
 * @class Grid
 * @brief Represents the computational grid for the CFD simulation
 *
 * The Grid class manages a 2D array of Cell objects, providing methods
 * for setting up the simulation domain, obstacles, and materials.
 */
class Grid {
private:
  // Grid dimensions
  size_t m_nx;
  size_t m_ny;

  // Grid origin.
  double m_origin_x;
  double m_origin_y;

  // Physical domain size
  double m_width;
  double m_height;

  // Cell size
  double m_dx;
  double m_dy;

  // Storage for cells
  std::vector<Cell> m_cells;

public:
  /**
   * @brief Constructor for Grid
   * @param nx Number of grid cells in x-direction
   * @param ny Number of grid cells in y-direction
   * @param width Physical width of the domain
   * @param height Physical height of the domain
   */
  Grid(size_t nx, size_t ny, double width, double height, double origin_x = 0.0,
       double origin_y = 0.0);

  std::shared_ptr<Grid> createWithUnits(size_t nx, size_t ny, double width,
                                        double height,
                                        const std::string &lengthUnit,
                                        double originX = 0.0,
                                        double originY = 0.0);

  // Index calculation for 1D array - inline for performance
  inline size_t index(size_t i, size_t j) const { return i + j * m_nx; }

  // Cell access methods with bounds checking in debug builds
  inline Cell &getCell(size_t i, size_t j) {
#ifdef DEBUG
    if (i >= m_nx || j >= m_ny) {
      throw std::out_of_range("Cell indices out of range");
    }
#endif
    return m_cells[index(i, j)];
  }

  const Cell &getCell(size_t i, size_t j) const {
#ifdef DEBUG
    if (i >= m_nx || j >= m_ny) {
      throw std::out_of_range("Cell indices out of range");
    }
#endif
    return m_cells[index(i, j)];
  }

  // Direct array access for high-performance loops
  inline Cell *getCellData() { return m_cells.data(); }
  inline const Cell *getCellData() const { return m_cells.data(); }

  // Grid properties
  inline size_t getNx() const { return m_nx; }
  inline size_t getNy() const { return m_ny; }
  inline double getDx() const { return m_dx; }
  inline double getDy() const { return m_dy; }
  inline double getWidth() const { return m_width; }
  inline double getHeight() const { return m_height; }

  // Get physical dimensions with unit conversion
  double getWidthWithUnits(const std::string &lengthUnit) const {
    return Units::convert(getWidth(), "m", lengthUnit);
  }

  double getHeightWithUnits(const std::string &lengthUnit) const {
    return Units::convert(getHeight(), "m", lengthUnit);
  }

  double getOriginXWithUnits(const std::string &lengthUnit) const {
    return Units::convert(getOriginX(), "m", lengthUnit);
  }

  double getOriginYWithUnits(const std::string &lengthUnit) const {
    return Units::convert(getOriginY(), "m", lengthUnit);
  }

  // Get cell size with unit conversion
  double getDxWithUnits(const std::string &lengthUnit) const {
    return Units::convert(getDx(), "m", lengthUnit);
  }

  double getDyWithUnits(const std::string &lengthUnit) const {
    return Units::convert(getDy(), "m", lengthUnit);
  }

  // Convert from grid indices to physical coordinates with unit conversion
  double physicalXWithUnits(size_t i, const std::string &lengthUnit) const {
    return Units::convert(physicalX(i), "m", lengthUnit);
  }

  double physicalYWithUnits(size_t j, const std::string &lengthUnit) const {
    return Units::convert(physicalY(j), "m", lengthUnit);
  }

  // Convert from physical coordinates with units to grid indices
  size_t gridIFromUnits(double x, const std::string &lengthUnit) const {
    double xMeters = Units::convert(x, lengthUnit, "m");
    return gridI(xMeters);
  }

  size_t gridJFromUnits(double y, const std::string &lengthUnit) const {
    double yMeters = Units::convert(y, lengthUnit, "m");
    return gridJ(yMeters);
  }

  // Convert from grid indices to physical coordinates
  inline double physicalX(size_t i) const { return m_origin_x + i * m_dx; }
  inline double physicalY(size_t j) const { return m_origin_y + j * m_dy; }
  inline double getOriginX() const { return m_origin_x; }
  inline double getOriginY() const { return m_origin_y; }

  // Convert from physical coordinates to grid indices
  inline size_t gridI(double x) const {
    size_t i = static_cast<size_t>((x - m_origin_x) / m_dx + 0.5);
    return std::min(i, m_nx - 1); // Clamp to valid range
  }

  inline size_t gridJ(double y) const {
    size_t j = static_cast<size_t>((y - m_origin_y) / m_dy + 0.5);
    return std::min(j, m_ny - 1); // Clamp to valid range
  }

  // Initialize grid with default material
  inline void initialize(Material *default_material) {
    if (!default_material) {
      throw std::invalid_argument("Default material cannot be null");
    }

    for (auto &cell : m_cells) {
      cell.setMaterial(default_material);
      cell.setPressure(0.0);
      cell.setVelocityU(0.0);
      cell.setVelocityV(0.0);
      cell.setTemperature(0.0);
    }
  }

  void initializeTemperature(double defaultTemp = 300.0);
  void setTemperature(size_t i, size_t j, double temp);
  double getTemperature(size_t i, size_t j) const;
  double interpolateTemperature(double x, double y) const;

  // Set material for a specific region
  void setMaterialRegion(size_t i_start, size_t i_end, size_t j_start,
                         size_t j_end, Material *material);

  void setMaterialRegionWithUnits(double min_x, double max_x, double min_y,
                                  double max_y, Material *material,
                                  const std::string &lengthUnit);

  // Define a circular obstacle
  void setCircularObstacle(double center_x, double center_y, double radius,
                           Material *material);

  /**
   * @brief Define a rectangular obstacle
   * @param min_x Minimum x-coordinate of rectangle
   * @param min_y Minimum y-coordinate of rectangle
   * @param max_x Maximum x-coordinate of rectangle
   * @param max_y Maximum y-coordinate of rectangle
   * @param material Material for the obstacle cells
   */
  void setRectangularObstacle(double min_x, double min_y, double max_x,
                              double max_y, Material *material);

  /**
   * @brief Define a square obstacle
   * @param center_x X-coordinate of square center
   * @param center_y Y-coordinate of square center
   * @param side_length Length of square sides
   * @param material Material for the obstacle cells
   */
  void setSquareObstacle(double center_x, double center_y, double side_length,
                         Material *material);

  /**
   * @brief Define a triangular obstacle
   * @param x1 X-coordinate of first vertex
   * @param y1 Y-coordinate of first vertex
   * @param x2 X-coordinate of second vertex
   * @param y2 Y-coordinate of second vertex
   * @param x3 X-coordinate of third vertex
   * @param y3 Y-coordinate of third vertex
   * @param material Material for the obstacle cells
   */
  void setTriangularObstacle(double x1, double y1, double x2, double y2,
                             double x3, double y3, Material *material);

  /**
   * @brief Define an elliptical obstacle
   * @param center_x X-coordinate of ellipse center
   * @param center_y Y-coordinate of ellipse center
   * @param radius_x Semi-major axis in x-direction
   * @param radius_y Semi-major axis in y-direction
   * @param rotation_angle Rotation angle in degrees
   * @param material Material for the obstacle cells
   */
  void setEllipticalObstacle(double center_x, double center_y, double radius_x,
                             double radius_y, double rotation_angle,
                             Material *material);

  /**
   * @brief Define an airfoil-shaped obstacle using NACA 4-digit parameters
   * @param leading_edge_x X-coordinate of the airfoil leading edge
   * @param leading_edge_y Y-coordinate of the airfoil leading edge
   * @param chord_length Length of the airfoil chord
   * @param angle_of_attack Angle of attack in degrees
   * @param naca_digits NACA 4-digit code (e.g., 0012, 2412)
   * @param material Material for the obstacle cells
   */
  void setAirfoilObstacle(double leading_edge_x, double leading_edge_y,
                          double chord_length, double angle_of_attack,
                          int naca_digits, Material *material);

  /**
   * @brief Define a polygonal obstacle
   * @param vertices Vector of (x,y) coordinates defining the polygon vertices
   * @param material Material for the obstacle cells
   */
  void
  setPolygonObstacle(const std::vector<std::pair<double, double>> &vertices,
                     Material *material);

  // Clear velocities across the grid (useful between simulation steps)
  inline void clearVelocities() {
    for (auto &cell : m_cells) {
      if (!cell.isObstacle()) { // Preserve zero velocity for obstacles
        cell.setVelocityU(0.0);
        cell.setVelocityV(0.0);
      }
    }
  }

  // Clear pressures across the grid
  inline void clearPressures() {
    for (auto &cell : m_cells) {
      cell.setPressure(0.0);
    }
  }

  // Utility methods for high-performance solvers

  // Get velocity field for visualization or analysis
  void getVelocityField(std::vector<double> &vx, std::vector<double> &vy) const;

  // Get pressure field for visualization or analysis
  std::vector<double> getPressureField() const;

  // Get pressure field for visualization or analysis
  std::vector<double> getTemperatureField() const;

  // Calculate divergence at a cell (useful for pressure solvers)
  double calculateDivergence(size_t i, size_t j) const;

  // Check if grid is properly initialized
  bool isInitialized() const {
    for (const auto &cell : m_cells) {
      if (cell.getMaterial() == nullptr) {
        return false;
      }
    }
    return true;
  }
};