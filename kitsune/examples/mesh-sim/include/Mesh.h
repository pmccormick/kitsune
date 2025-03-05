/**
 * ====================================================================
 * Mesh Class - Core Implementation for CFD Simulations
 * ====================================================================
 *
 * This implementation focuses on the core grid functionality with a flat
 * memory layout for optimal performance, clean boundary condition handling,
 * and consistent coordinate conversions.
 */
#pragma once

#include <array>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "BoundaryClass.h"
#include "Cell.h"
#include "Material.h"
#include "Units.h"

// Forward declarations
class DirichletBoundary;
class NeumannBoundary;
class InflowBoundary;
class NoSlipBoundary;
class SlipBoundary;
class PeriodicBoundary;

/**
 * @class Mesh
 * @brief Core structured grid implementation for CFD simulations
 *
 * Features:
 * - Flat memory layout for cache-friendly access and potential GPU acceleration
 * - Comprehensive boundary condition handling
 * - Consistent coordinate conversions
 * - Built for extensibility through separate specialized modules
 */
class Mesh {
public:
  /**
   * @enum BoundaryLocation
   * @brief Defines the standard locations of domain boundaries
   */
  enum class BoundaryLocation {
    LEFT,   ///< Left (min-X) boundary
    RIGHT,  ///< Right (max-X) boundary
    BOTTOM, ///< Bottom (min-Y) boundary
    TOP,    ///< Top (max-Y) boundary
    ALL     ///< All domain boundaries
  };

  /**
   * @enum FieldType
   * @brief Types of field data that can be extracted
   */
  enum class FieldType {
    VELOCITY_X,    ///< X-component of velocity
    VELOCITY_Y,    ///< Y-component of velocity
    PRESSURE,      ///< Pressure field
    TEMPERATURE,   ///< Temperature field
    DENSITY,       ///< Density field
    MATERIAL_ID,   ///< Material identifier
    BOUNDARY_TYPE, ///< Boundary condition type
    CELL_TYPE      ///< Cell type (fluid, solid, boundary)
  };

  /**
   * @brief Constructor for Mesh
   * @param nx Number of grid cells in x-direction
   * @param ny Number of grid cells in y-direction
   * @param width Physical width of the domain (meters)
   * @param height Physical height of the domain (meters)
   * @param origin_x X-coordinate of domain origin (meters)
   * @param origin_y Y-coordinate of domain origin (meters)
   */
  Mesh(size_t nx, size_t ny, double width, double height, double origin_x = 0.0,
       double origin_y = 0.0);

  /**
   * @brief Create a grid with dimensions specified in non-SI units
   * @param nx Number of grid cells in x-direction
   * @param ny Number of grid cells in y-direction
   * @param width Physical width of the domain
   * @param height Physical height of the domain
   * @param lengthUnit Unit for length measurements (e.g., "m", "ft", "in")
   * @param origin_x X-coordinate of domain origin
   * @param origin_y Y-coordinate of domain origin
   * @return Shared pointer to the created grid
   */
  static std::shared_ptr<Mesh> createWithUnits(size_t nx, size_t ny,
                                               double width, double height,
                                               const std::string &lengthUnit,
                                               double origin_x = 0.0,
                                               double origin_y = 0.0);

  /**
   * @brief Destructor
   */
  ~Mesh();

  //==== Grid Dimension Accessors ====

  /**
   * @brief Get the number of cells in the x-direction
   * @return Number of cells in x-direction
   */
  size_t getNx() const { return m_nx; }

  /**
   * @brief Get the number of cells in the y-direction
   * @return Number of cells in y-direction
   */
  size_t getNy() const { return m_ny; }

  /**
   * @brief Get the physical width of the domain
   * @return Width in meters
   */
  double getWidth() const { return m_width; }

  /**
   * @brief Get the physical height of the domain
   * @return Height in meters
   */
  double getHeight() const { return m_height; }

  /**
   * @brief Get the cell size in the x-direction
   * @return Cell size in meters
   */
  double getDx() const { return m_dx; }

  /**
   * @brief Get the cell size in the y-direction
   * @return Cell size in meters
   */
  double getDy() const { return m_dy; }

  /**
   * @brief Get the x-coordinate of the domain origin
   * @return Origin x-coordinate in meters
   */
  double getOriginX() const { return m_origin_x; }

  /**
   * @brief Get the y-coordinate of the domain origin
   * @return Origin y-coordinate in meters
   */
  double getOriginY() const { return m_origin_y; }

  //==== Unit Conversion Methods ====

  /**
   * @brief Get the physical width of the domain with unit conversion
   * @param lengthUnit Unit for the return value (e.g., "m", "ft", "in")
   * @return Width in specified units
   */
  double getWidthWithUnits(const std::string &lengthUnit) const {
    return Units::convert(getWidth(), "m", lengthUnit);
  }

  /**
   * @brief Get the physical height of the domain with unit conversion
   * @param lengthUnit Unit for the return value (e.g., "m", "ft", "in")
   * @return Height in specified units
   */
  double getHeightWithUnits(const std::string &lengthUnit) const {
    return Units::convert(getHeight(), "m", lengthUnit);
  }

  /**
   * @brief Get the cell size in x-direction with unit conversion
   * @param lengthUnit Unit for the return value (e.g., "m", "ft", "in")
   * @return Cell size in specified units
   */
  double getDxWithUnits(const std::string &lengthUnit) const {
    return Units::convert(getDx(), "m", lengthUnit);
  }

  /**
   * @brief Get the cell size in y-direction with unit conversion
   * @param lengthUnit Unit for the return value (e.g., "m", "ft", "in")
   * @return Cell size in specified units
   */
  double getDyWithUnits(const std::string &lengthUnit) const {
    return Units::convert(getDy(), "m", lengthUnit);
  }

  /**
   * @brief Get the origin x-coordinate with unit conversion
   * @param lengthUnit Unit for the return value (e.g., "m", "ft", "in")
   * @return Origin x-coordinate in specified units
   */
  double getOriginXWithUnits(const std::string &lengthUnit) const {
    return Units::convert(getOriginX(), "m", lengthUnit);
  }

  /**
   * @brief Get the origin y-coordinate with unit conversion
   * @param lengthUnit Unit for the return value (e.g., "m", "ft", "in")
   * @return Origin y-coordinate in specified units
   */
  double getOriginYWithUnits(const std::string &lengthUnit) const {
    return Units::convert(getOriginY(), "m", lengthUnit);
  }

  //==== Cell Access Methods ====

  /**
   * @brief Get reference to a cell at the specified grid indices
   * @param i Grid index in x-direction
   * @param j Grid index in y-direction
   * @return Reference to the cell
   */
  Cell &getCell(size_t i, size_t j);

  /**
   * @brief Get const reference to a cell at the specified grid indices
   * @param i Grid index in x-direction
   * @param j Grid index in y-direction
   * @return Const reference to the cell
   */
  const Cell &getCell(size_t i, size_t j) const;

  /**
   * @brief Get direct pointer to the cell data array for high-performance
   * access
   * @return Pointer to the first cell in the array
   */
  Cell *getCellData() { return m_cells.data(); }

  /**
   * @brief Get direct const pointer to the cell data array for high-performance
   * access
   * @return Const pointer to the first cell in the array
   */
  const Cell *getCellData() const { return m_cells.data(); }

  /**
   * @brief Get total number of cells in the mesh
   * @return Total cell count
   */
  size_t getCellCount() const { return m_cells.size(); }

  //==== Coordinate Conversion Methods ====

  /**
   * @brief Convert grid index to physical x-coordinate
   * @param i Grid index in x-direction
   * @return Physical x-coordinate in meters
   */
  double physicalX(size_t i) const { return m_origin_x + i * m_dx; }

  /**
   * @brief Convert grid index to physical y-coordinate
   * @param j Grid index in y-direction
   * @return Physical y-coordinate in meters
   */
  double physicalY(size_t j) const { return m_origin_y + j * m_dy; }

  /**
   * @brief Convert grid index to physical x-coordinate with unit conversion
   * @param i Grid index in x-direction
   * @param lengthUnit Unit for the return value (e.g., "m", "ft", "in")
   * @return Physical x-coordinate in specified units
   */
  double physicalXWithUnits(size_t i, const std::string &lengthUnit) const {
    return Units::convert(physicalX(i), "m", lengthUnit);
  }

  /**
   * @brief Convert grid index to physical y-coordinate with unit conversion
   * @param j Grid index in y-direction
   * @param lengthUnit Unit for the return value (e.g., "m", "ft", "in")
   * @return Physical y-coordinate in specified units
   */
  double physicalYWithUnits(size_t j, const std::string &lengthUnit) const {
    return Units::convert(physicalY(j), "m", lengthUnit);
  }

  /**
   * @brief Convert physical x-coordinate to grid index
   * @param x Physical x-coordinate in meters
   * @return Grid index in x-direction
   */
  size_t gridI(double x) const {
    size_t i = static_cast<size_t>((x - m_origin_x) / m_dx + 0.5);
    return std::min(i, m_nx - 1); // Clamp to valid range
  }

  /**
   * @brief Convert physical y-coordinate to grid index
   * @param y Physical y-coordinate in meters
   * @return Grid index in y-direction
   */
  size_t gridJ(double y) const {
    size_t j = static_cast<size_t>((y - m_origin_y) / m_dy + 0.5);
    return std::min(j, m_ny - 1); // Clamp to valid range
  }

  /**
   * @brief Convert physical x-coordinate with units to grid index
   * @param x Physical x-coordinate in specified units
   * @param lengthUnit Unit of the x value (e.g., "m", "ft", "in")
   * @return Grid index in x-direction
   */
  size_t gridIFromUnits(double x, const std::string &lengthUnit) const {
    double xMeters = Units::convert(x, lengthUnit, "m");
    return gridI(xMeters);
  }

  /**
   * @brief Convert physical y-coordinate with units to grid index
   * @param y Physical y-coordinate in specified units
   * @param lengthUnit Unit of the y value (e.g., "m", "ft", "in")
   * @return Grid index in y-direction
   */
  size_t gridJFromUnits(double y, const std::string &lengthUnit) const {
    double yMeters = Units::convert(y, lengthUnit, "m");
    return gridJ(yMeters);
  }

  //==== Core Initialization Methods ====

  /**
   * @brief Initialize the mesh with a default material
   * @param defaultMaterial Material to assign to all cells
   */
  void initialize(std::shared_ptr<Material> defaultMaterial);

  //==== Boundary Condition Management ====

  /**
   * @brief Set boundary condition for a standard boundary location
   * @param location Standard boundary location (LEFT, RIGHT, TOP, BOTTOM, ALL)
   * @param boundary Shared pointer to the boundary condition
   */
  void setBoundaryCondition(BoundaryLocation location,
                            std::shared_ptr<BoundaryClass> boundary);

  /**
   * @brief Set boundary condition for a custom region
   * @param i_start Starting grid index in x-direction
   * @param i_end Ending grid index in x-direction (inclusive)
   * @param j_start Starting grid index in y-direction
   * @param j_end Ending grid index in y-direction (inclusive)
   * @param boundary Shared pointer to the boundary condition
   */
  void setBoundaryConditionRegion(size_t i_start, size_t i_end, size_t j_start,
                                  size_t j_end,
                                  std::shared_ptr<BoundaryClass> boundary);

  /**
   * @brief Apply all boundary conditions to update boundary cells
   * @param dt Time step size
   */
  void applyBoundaryConditions(double dt = 0.0);

  /**
   * @brief Get the array of boundary cells for a standard location
   * @param location Boundary location
   * @return Vector of pointers to cells at the specified boundary
   */
  std::vector<Cell *> getBoundaryCells(BoundaryLocation location);

  /**
   * @brief Get the array of boundary conditions
   * @return Map of boundary names to boundary condition objects
   */
  const std::unordered_map<std::string, std::shared_ptr<BoundaryClass>> &
  getBoundaryConditions() const {
    return m_boundaryConditions;
  }

  //==== Core Helper Methods ====

  /**
   * @brief Get neighboring cells for a specific cell (N, E, S, W order)
   * @param i Grid index in x-direction
   * @param j Grid index in y-direction
   * @return Array of pointers to the neighboring cells, null if outside grid
   */
  std::array<Cell *, 4> getNeighbors(size_t i, size_t j);

  /**
   * @brief Get neighboring cells for boundary conditions
   * @param i Grid index in x-direction
   * @param j Grid index in y-direction
   * @return Vector of pointers to internal (non-boundary) neighboring cells
   */
  std::vector<Cell *> getBoundaryNeighbors(size_t i, size_t j);

  //==== Basic Field Access Methods ====

  /**
   * @brief Get a field of data from the grid (primary fields only)
   * @param fieldType Type of field to extract
   * @return Vector containing the field data
   */
  std::vector<double> getField(FieldType fieldType) const;

  /**
   * @brief Get velocity field components
   * @param vx Output vector for x-velocity components
   * @param vy Output vector for y-velocity components
   */
  void getVelocityField(std::vector<double> &vx, std::vector<double> &vy) const;

  /**
   * @brief Get the pressure field
   * @return Vector of pressure values in Pascal
   */
  std::vector<double> getPressureField() const;

  /**
   * @brief Get the temperature field
   * @return Vector of temperature values in Kelvin
   */
  std::vector<double> getTemperatureField() const;

  /**
   * @brief Get the density field
   * @return Vector of density values in kg/m³
   */
  std::vector<double> getDensityField() const;

  //==== Cell Type Marking Methods ====

  /**
   * @brief Set a cell as an obstacle
   * @param i Grid index in x-direction
   * @param j Grid index in y-direction
   * @param material Material for the obstacle
   */
  void setCellAsObstacle(size_t i, size_t j,
                         std::shared_ptr<Material> material);

  /**
   * @brief Set a cell as a boundary
   * @param i Grid index in x-direction
   * @param j Grid index in y-direction
   * @param boundary Boundary condition to apply (optional)
   */
  void setCellAsBoundary(size_t i, size_t j,
                         std::shared_ptr<BoundaryClass> boundary = nullptr);

  /**
   * @brief Set a cell as fluid
   * @param i Grid index in x-direction
   * @param j Grid index in y-direction
   * @param material Material for the fluid (optional)
   */
  void setCellAsFluid(size_t i, size_t j,
                      std::shared_ptr<Material> material = nullptr);

  //==== Friend Declarations ====

  // Declare relevant external modules as friends to allow efficient access
  // This helps avoid unnecessary accessor/mutator methods
  // friend class ObstacleGenerator; // If we create these as classes
  // friend class GridVisualizer;

private:
  // Grid dimensions
  size_t m_nx;
  size_t m_ny;

  // Physical domain size in meters
  double m_width;
  double m_height;

  // Origin coordinates in meters
  double m_origin_x;
  double m_origin_y;

  // Cell sizes in meters
  double m_dx;
  double m_dy;

  // Storage for cells in flat array
  std::vector<Cell> m_cells;

  // Boundary condition mapping
  std::unordered_map<std::string, std::shared_ptr<BoundaryClass>>
      m_boundaryConditions;

  // Helper to compute 1D index from 2D indices
  inline size_t index(size_t i, size_t j) const { return i + j * m_nx; }
};
