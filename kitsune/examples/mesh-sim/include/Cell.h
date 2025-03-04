/**
 * ====================================================================
 * Cell Class - CFD Implementation Documentation
 * ====================================================================
 *
 * The Cell class represents a discrete cell within the computational grid for
 * CFD simulations. It manages physical properties (temperature, pressure,
 * density), flow variables (velocity components), boundary conditions, and
 * integration with material properties.
 *
 * Relationship with Grid and Materials:
 * ------------------------------------
 * - Grid Integration:
 *   Cells are the fundamental components managed by the Grid class. The Cell
 * implementation provides dedicated methods for Grid compatibility
 * (setBoundary, setObstacle, isBoundary, isObstacle) to support the Grid's
 * domain representation and boundary handling.
 *
 * - Material Relationship:
 *   Each Cell maintains a reference to a Material object (via
 *   std::shared_ptr<Material>) which defines the physical properties of the
 *   fluid/solid within the cell. The material association determines how the
 * cell behaves during simulation, particularly for properties like viscosity,
 *   thermal conductivity, and equation of state.
 *
 * Primary Use Cases:
 * ----------------
 * 1. Finite Volume Implementation
 *    - Stores primary flow variables (pressure, velocity, temperature) at cell
 * centers
 *    - Maintains vertex velocities for staggered grid arrangements
 *    - Supports both fixed and computed property values via PropertyType enum
 *
 * 2. Boundary Condition Management
 *    - Supports multiple boundary types via CellFlags (IS_INLET, IS_OUTLET,
 *      IS_WALL, IS_SYMMETRY)
 *    - Implements 'fixed' cells for enforcing boundary values
 *    - Differentiates between FLUID, SOLID, and BOUNDARY cell types
 *
 * 3. Units and Physical Consistency
 *    - Provides comprehensive unit conversion support for all physical
 *      properties
 *    - Enforces valid ranges for physical quantities like temperature,
 *      pressure, and density
 *    - Maintains SI units internally (K, Pa, kg/m³) with conversion helpers
 *
 * 4. Performance-Optimized Property Access
 *    - Implements three-tiered property storage for optimal access patterns:
 *      a) Core properties (temperature, pressure) as direct member variables
 *      b) Common numerical properties via fixed-size array (PropertyType enum)
 *      c) Dynamic properties via string-indexed map for flexibility
 *    - Bit-packed boolean flags for memory-efficient state tracking
 *
 * Key Cell Properties:
 * ------------------
 * - Physical State: Temperature (K), Pressure (Pa), Density (kg/m³)
 * - Flow Variables: Velocity components at cell center and vertices
 * - Numerical Properties: Vorticity, stream function, kinetic energy, etc.
 * - Boundary Information: Type flags (inlet, outlet, wall, symmetry)
 *
 * Cell Types and Classification:
 * ----------------------------
 *    FLUID Cells: Regular computational cells where flow equations are solved
 *    SOLID Cells: Obstacle cells representing solid objects in the flow
 *    BOUNDARY Cells: Domain boundary cells with specified conditions
 *
 * ====================================================================
 * PropertyType Usage Guide
 * ====================================================================
 *
 * The PropertyType enum in the Cell class is designed to provide efficient
 * access to commonly used numerical properties in CFD simulations. These
 * properties are typically calculated during each timestep and need to be
 * accessed frequently.
 *
 * The PropertyType approach offers three key benefits:
 *
 * 1. PERFORMANCE OPTIMIZATION:
 *    PropertyType uses a fixed-size array with enum indexing for O(1) access.
 *    This is significantly faster than the string-based dynamic property lookup
 *    which requires hashing and map traversal.
 *
 * 2. MEMORY EFFICIENCY:
 *    Properties are stored in a fixed-size array allocated at initialization,
 *    avoiding memory fragmentation from dynamic allocations.
 *
 * 3. TYPE SAFETY:
 *    Using an enum instead of string identifiers provides compile-time checking
 *    and prevents typos or naming inconsistencies.
 *
 *    Using an enum instead of string identifiers provides compile-time checking
 *    and prevents typos or naming inconsistencies.
 *
 * Available PropertyTypes:
 * -------------------------
 * - VORTICITY: Measures local rotation in the flow field (ω = ∇ × v)
 * - STREAM_FUNCTION: Used for streamline visualization in 2D flows
 * - KINETIC_ENERGY: Local kinetic energy per unit mass (0.5 * |v|²)
 * - DIVERGENCE: Velocity divergence (∇·v), should be ~0 for incompressible flow
 * - PRESSURE_CORRECTION: Term used in pressure-correction algorithms
 * - HEAT_FLUX_X/Y: Heat transfer rate in x/y directions
 * - SHEAR_STRESS: Local fluid shear stress
 * - WALL_DISTANCE: Distance to nearest wall (for turbulence models)
 *
 * Usage Pattern:
 * -------------
 * 1. During simulation, calculate derived properties like vorticity
 * 2. Store using cell.setProperty(PropertyType::VORTICITY, value)
 * 3. Access using cell.getProperty(PropertyType::VORTICITY)
 *
 * When to use PropertyType vs. Dynamic Properties:
 * ----------------------------------------------
 * - Use PropertyType for common, performance-critical properties
 * - Use dynamic properties (string-based) for:
 *   a) Temporary or rarely accessed values
 *   b) User-defined or custom properties not in the enum
 *   c) Properties with variable/runtime-defined names
 *
 * Structure Diagram:
 * ----------------
 *    +---------------+
 *    | Cell          |
 *    |---------------|
 *    | m_type        | ◄── FLUID/SOLID/BOUNDARY
 *    | m_temperature |
 *    | m_pressure    |      +----------+
 *    | m_density     |      | Material |
 *    | m_velocity_x/y| ◄────┘ properties
 *    | m_vertices    |
 *    | m_fixedProps  |
 *    | m_flags       |
 *    | m_dynProps    |
 *    +---------------+
 *
 * Performance Considerations:
 * -------------------------
 * - Access common properties via PropertyType enum for optimal performance
 * - Use CellFlag enum for boolean state tracking rather than dynamic properties
 * - Consider material property lookups in performance-critical code sections
 * - Dynamic properties provide flexibility but with higher access overhead
 *
 * Usage Notes:
 * ----------
 * - For boundary conditions, ensure both cell type and appropriate flags are
 *   set
 * - When setting cell types, appropriate flags are automatically managed
 * - Use the Units conversion helpers for consistent physical quantities
 * - Reset() preserves cell type and material but clears all other state
 *
 * LIMITATIONS:
 * -----------
 * 1. CURRENT IMPLEMENTATION:
 *    - Supports only 2D structured grids (rectangular cells with 4 vertices)
 *    - No direct support for higher-order interpolation schemes
 *    - Assumes cells maintain consistent connectivity throughout simulation
 *
 * 2. MATERIAL INTEGRATION:
 *    - Material properties are referenced but not automatically updated
 *    - Changes to material require explicit update in simulation code
 *    - No automatic temperature/pressure dependency for material properties
 *
 * 3. MEMORY CONSIDERATIONS:
 *    - Dynamic property map may cause memory fragmentation with many properties
 *    - Shared pointer to Material increases reference count overhead
 *
 * 4. GRID INTEGRATION:
 *    - Cell-to-cell connectivity must be managed externally by Grid class
 *    - No direct neighbor awareness within the Cell implementation
 *    - Grid boundary detection requires explicit flag setting
 */

#pragma once

#include "Units.h"
#include <array>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declaration
class Material;
class Grid;
class BoundaryClass;

/**
 * @class Cell
 * @brief Represents a single cell in the computational grid for CFD simulation
 *
 * This class stores physical properties like temperature, pressure, velocity at
 * the cell center. It supports both the original functionality and the
 * requirements of the Grid class.
 */
class Cell {
public:
  /**
   * @enum CellType
   * @brief Defines the possible types of cells in the simulation
   */
  enum class CellType {
    FLUID,   ///< Regular fluid cell
    SOLID,   ///< Solid obstacle cell (e.g., cylinder)
    BOUNDARY ///< Boundary cell
  };

  /**
   * @enum VertexPosition
   * @brief Defines the relative positions of cell vertices
   */
  enum class VertexPosition {
    NORTHWEST = 0, // Top left
    NORTHEAST = 1, // Top right
    SOUTHEAST = 2, // Bottom right
    SOUTHWEST = 3  // Bottom left
  };

  /**
   * @enum PropertyType
   * @brief Defines the common numerical properties that are frequently accessed
   */
  enum class PropertyType {
    VORTICITY = 0,       ///< Curl of velocity field (rotation)
    STREAM_FUNCTION = 1, ///< Stream function value (for visualization)
    KINETIC_ENERGY = 2,  ///< Kinetic energy per unit mass
    DIVERGENCE = 3,      ///< Velocity divergence (should be near zero for
                         ///< incompressible flow)
    PRESSURE_CORRECTION =
        4,             ///< Pressure correction term in SIMPLE/PISO algorithms
    HEAT_FLUX_X = 5,   ///< Heat flux in x-direction
    HEAT_FLUX_Y = 6,   ///< Heat flux in y-direction
    SHEAR_STRESS = 7,  ///< Local shear stress
    WALL_DISTANCE = 8, ///< Distance to nearest wall (for turbulence models)
    COUNT              ///< Keep last - used for array sizing
  };

  /**
   * @enum CellFlag
   * @brief Boolean flags for various cell states and behaviors
   */
  enum class CellFlag {
    IS_INLET = 0x00000001,    //< Cell is part of an inlet boundary
    IS_OUTLET = 0x00000002,   ///< Cell is part of an outlet boundary
    IS_WALL = 0x00000004,     ///< Cell is part of a wall boundary
    IS_SYMMETRY = 0x00000008, ///< Cell is part of a symmetry boundary
    IS_BOUNDARY = 0x00000010, ///< Cell is a boundary cell (Grid compatibility)
    IS_OBSTACLE = 0x00000020, ///< Cell is an obstacle (for Grid compatibility)
    COUNT                     ///< Keep last - used for bit field sizing
  };

  /**
   * @struct Vertex
   * @brief Data structure for cell vertex information
   */
  struct Vertex {
    double vx = 0.0; ///< x-component of velocity
    double vy = 0.0; ///< y-component of velocity
  };

  // Constructor should be updated to include a reference to the grid
  Cell(Grid *grid = nullptr);

  /**
   * @brief Parameterized constructor
   * @param type The type of cell (fluid, solid, boundary)
   */
  explicit Cell(CellType type, Grid *grid = nullptr);

  void setBoundaryCondition(std::shared_ptr<BoundaryClass> boundary) {
    m_boundaryCondition = boundary;
  }

  std::shared_ptr<BoundaryClass> getBoundaryCondition() const {
    return m_boundaryCondition;
  }

  bool hasBoundaryCondition() const { return m_boundaryCondition != nullptr; }

  Grid *getGrid() const { return m_grid; }

  /**
   * @brief Get the cell type
   * @return The cell type
   */
  CellType getType() const;

  /**
   * @brief Set the cell type
   * @param type The cell type to set
   */
  void setType(CellType type);

  /**
   * @brief Get the temperature at the cell center
   * @return The temperature value (K)
   */
  double getTemperature() const;

  /**
   * @brief Set the temperature at the cell center
   * @param temperature The temperature value to set (K)
   */
  void setTemperature(double temperature);

  /**
   * @brief Set temperature with unit conversion
   * @param temperature Temperature value in specified units
   * @param unit Unit string (e.g., "K", "C", "F")
   */
  void setTemperatureWithUnits(double temperature, const std::string &unit) {
    double kelvin = Units::convert(temperature, unit, "K");
    setTemperature(kelvin);
  }

  /**
   * @brief Get temperature with unit conversion
   * @param unit Unit string to convert to (e.g., "K", "C", "F")
   * @return Temperature in requested units
   */
  double getTemperatureWithUnits(const std::string &unit) const {
    return Units::convert(getTemperature(), "K", unit);
  }

  /**
   * @brief Get the pressure at the cell center
   * @return The pressure value (Pa)
   */
  double getPressure() const;

  /**
   * @brief Set the pressure at the cell center
   * @param pressure The pressure value to set (Pa)
   */
  void setPressure(double pressure);

  /**
   * @brief Set pressure with unit conversion
   * @param pressure Pressure value in specified units
   * @param unit Unit string (e.g., "Pa", "bar", "atm", "psi")
   */
  void setPressureWithUnits(double pressure, const std::string &unit) {
    double pascal = Units::convert(pressure, unit, "Pa");
    setPressure(pascal);
  }

  /**
   * @brief Get pressure with unit conversion
   * @param unit Unit string to convert to (e.g., "Pa", "bar", "atm", "psi")
   * @return Pressure in requested units
   */
  double getPressureWithUnits(const std::string &unit) const {
    return Units::convert(getPressure(), "Pa", unit);
  }

  /**
   * @brief Get the density at the cell center
   * @return The density value (kg/m³)
   */
  double getDensity() const;

  /**
   * @brief Set the density at the cell center
   * @param density The density value to set (kg/m³)
   */
  void setDensity(double density);

  /**
   * @brief Set density with unit conversion
   * @param density Density value in specified units
   * @param unit Unit string (e.g., "kg/m³", "lb/ft³")
   */
  void setDensityWithUnits(double density, const std::string &unit) {
    double kgm3 = Units::convert(density, unit, "kg/m³");
    setDensity(kgm3);
  }

  /**
   * @brief Get density with unit conversion
   * @param unit Unit string to convert to (e.g., "kg/m³", "lb/ft³")
   * @return Density in requested units
   */
  double getDensityWithUnits(const std::string &unit) const {
    return Units::convert(getDensity(), "kg/m³", unit);
  }

  /**
   * @brief Get the material assigned to this cell
   * @return Pointer to the material
   */
  std::shared_ptr<Material> getMaterial() const;

  /**
   * @brief Set the material for this cell
   * @param material Pointer to the material
   */
  void setMaterial(std::shared_ptr<Material> material);

  /**
   * @brief Set the material for this cell (raw pointer version for Grid
   * compatibility)
   * @param material Pointer to the material
   */
  void setMaterial(Material *material);

  /**
   * @brief Check if the cell has fixed values (boundary condition)
   * @return True if the cell has fixed values
   */
  bool isFixed() const;

  /**
   * @brief Set whether the cell has fixed values
   * @param fixed True to set the cell as fixed
   */
  void setFixed(bool fixed);

  /**
   * @brief Get velocity components at a specific vertex
   * @param position The vertex position
   * @return Pair of (vx, vy) velocity components
   */
  std::pair<double, double> getVertexVelocity(VertexPosition position) const;

  /**
   * @brief Set velocity components at a specific vertex
   * @param position The vertex position
   * @param vx The x-component of velocity
   * @param vy The y-component of velocity
   */
  void setVertexVelocity(VertexPosition position, double vx, double vy);

  /**
   * @brief Set velocity components with unit conversion
   * @param vx X-velocity in specified units
   * @param vy Y-velocity in specified units
   * @param unit Unit string (e.g., "m/s", "mph", "knot")
   */
  void setVelocityWithUnits(double vx, double vy, const std::string &unit) {
    double vx_mps = Units::convert(vx, unit, "m/s");
    double vy_mps = Units::convert(vy, unit, "m/s");
    setVelocityX(vx_mps);
    setVelocityY(vy_mps);
  }

  /**
   * @brief Get x-velocity with unit conversion
   * @param unit Unit string to convert to (e.g., "m/s", "mph", "knot")
   * @return X-velocity in requested units
   */
  double getVelocityXWithUnits(const std::string &unit) const {
    return Units::convert(getVelocityX(), "m/s", unit);
  }

  /**
   * @brief Get y-velocity with unit conversion
   * @param unit Unit string to convert to (e.g., "m/s", "mph", "knot")
   * @return Y-velocity in requested units
   */
  double getVelocityYWithUnits(const std::string &unit) const {
    return Units::convert(getVelocityY(), "m/s", unit);
  }

  /**
   * @brief Set vertex velocity with unit conversion
   * @param position The vertex position
   * @param vx X-velocity in specified units
   * @param vy Y-velocity in specified units
   * @param unit Unit string (e.g., "m/s", "mph", "knot")
   */
  void setVertexVelocityWithUnits(VertexPosition position, double vx, double vy,
                                  const std::string &unit) {
    double vx_mps = Units::convert(vx, unit, "m/s");
    double vy_mps = Units::convert(vy, unit, "m/s");
    setVertexVelocity(position, vx_mps, vy_mps);
  }

  /**
   * @brief Get vertex velocity with unit conversion
   * @param position The vertex position
   * @param unit Unit string to convert to (e.g., "m/s", "mph", "knot")
   * @return Pair of (vx, vy) velocity components in requested units
   */
  std::pair<double, double>
  getVertexVelocityWithUnits(VertexPosition position,
                             const std::string &unit) const {
    auto [vx_mps, vy_mps] = getVertexVelocity(position);
    return {Units::convert(vx_mps, "m/s", unit),
            Units::convert(vy_mps, "m/s", unit)};
  }

  /**
   * @brief Get reference to a vertex
   * @param position The vertex position
   * @return Reference to the vertex
   */
  Vertex &getVertex(VertexPosition position);

  /**
   * @brief Get const reference to a vertex
   * @param position The vertex position
   * @return Const reference to the vertex
   */
  const Vertex &getVertex(VertexPosition position) const;

  /**
   * @brief Set a fixed property value (fast access)
   * @param type The property type
   * @param value The property value
   */
  void setProperty(PropertyType type, double value);

  /**
   * @brief Get a fixed property value (fast access)
   * @param type The property type
   * @param defaultValue Default value if not set
   * @return The property value
   */
  double getProperty(PropertyType type) const;

  // Vorticity - local rotation in the flow field (1/s)
  double getVorticity() const { return getProperty(PropertyType::VORTICITY); }
  void setVorticity(double value) {
    setProperty(PropertyType::VORTICITY, value);
  }

  // Stream function for 2D flow visualization (m²/s)
  double getStreamFunction() const {
    return getProperty(PropertyType::STREAM_FUNCTION);
  }
  void setStreamFunction(double value) {
    setProperty(PropertyType::STREAM_FUNCTION, value);
  }

  // Kinetic energy per unit mass (J/kg or m²/s²)
  double getKineticEnergy() const {
    return getProperty(PropertyType::KINETIC_ENERGY);
  }
  void setKineticEnergy(double value) {
    setProperty(PropertyType::KINETIC_ENERGY, value);
  }

  // Velocity divergence (1/s) - should be ~0 for incompressible flow
  double getDivergence() const { return getProperty(PropertyType::DIVERGENCE); }
  void setDivergence(double value) {
    setProperty(PropertyType::DIVERGENCE, value);
  }

  // Pressure correction term for SIMPLE/PISO algorithms (Pa)
  double getPressureCorrection() const {
    return getProperty(PropertyType::PRESSURE_CORRECTION);
  }
  void setPressureCorrection(double value) {
    setProperty(PropertyType::PRESSURE_CORRECTION, value);
  }

  // Heat flux components (W/m²)
  double getHeatFluxX() const { return getProperty(PropertyType::HEAT_FLUX_X); }
  void setHeatFluxX(double value) {
    setProperty(PropertyType::HEAT_FLUX_X, value);
  }

  double getHeatFluxY() const { return getProperty(PropertyType::HEAT_FLUX_Y); }
  void setHeatFluxY(double value) {
    setProperty(PropertyType::HEAT_FLUX_Y, value);
  }

  // Local shear stress (Pa)
  double getShearStress() const {
    return getProperty(PropertyType::SHEAR_STRESS);
  }
  void setShearStress(double value) {
    setProperty(PropertyType::SHEAR_STRESS, value);
  }

  // Distance to nearest wall for turbulence models (m)
  double getWallDistance() const {
    return getProperty(PropertyType::WALL_DISTANCE);
  }
  void setWallDistance(double value) {
    setProperty(PropertyType::WALL_DISTANCE, value);
  }

  /**
   * @brief Set a boolean flag
   * @param flag The flag to set
   * @param value The flag value (true/false)
   */
  void setFlag(CellFlag flag, bool value);

  /**
   * @brief Get a boolean flag value
   * @param flag The flag to check
   * @return The flag value
   */
  bool getFlag(CellFlag flag) const;

  /**
   * @brief Set a dynamic property (less common properties, slower access)
   * @param name The property name
   * @param value The property value
   */
  void setDynamicProperty(const std::string &name, double value);

  /**
   * @brief Get a dynamic property value (slower access)
   * @param name The property name
   * @param defaultValue Default value if not found
   * @return The property value
   */
  double getDynamicProperty(const std::string &name,
                            double defaultValue = 0.0) const;

  /**
   * @brief Reset the cell to default values
   */
  void reset();

  // Additional methods for Grid compatibility

  /**
   * @brief Set whether the cell is a boundary
   * @param isBoundary True if cell is a boundary
   */
  void setBoundary(bool isBoundary);

  /**
   * @brief Check if the cell is a boundary
   * @return True if the cell is a boundary
   */
  bool isBoundary() const;

  /**
   * @brief Set whether the cell is an obstacle
   * @param isObstacle True if cell is an obstacle
   */
  void setObstacle(bool isObstacle);

  /**
   * @brief Check if the cell is an obstacle
   * @return True if the cell is an obstacle
   */
  bool isObstacle() const;

  /**
   * @brief Get the x-component of velocity at cell center
   * @return X-velocity
   */
  double getVelocityX() const;

  /**
   * @brief Get the y-component of velocity at cell center
   * @return Y-velocity
   */
  double getVelocityY() const;

  /**
   * @brief Set the x-component of velocity at cell center
   * @param vx X-velocity to set
   */
  void setVelocityX(double vx);

  /**
   * @brief Set the y-component of velocity at cell center
   * @param vy Y-velocity to set
   */
  void setVelocityY(double vy);

  /**
   * @brief Function signature for property computation
   * @param cell The cell to compute properties for
   * @param neighbors Optional array of neighboring cells
   */
  using PropertyComputeFunction =
      std::function<void(Cell &, const std::array<Cell *, 4> *)>;

  /**
   * @brief Register a custom property computation function for a specific
   * property
   * @param type The property type to register for
   * @param computeFunc The function to compute this property
   */
  static void registerPropertyComputation(PropertyType type,
                                          PropertyComputeFunction computeFunc);

  /**
   * @brief Compute derived properties based on current state using registered
   * computation functions
   * @param neighbors Optional array of neighboring cells for gradient-based
   * properties
   */
  void  computeDerivedProperties(const std::array<Cell *, 4> *neighbors = nullptr);

  /**
   * @brief Get the computation function for a specific property
   * @param type The property type to get the computation for
   * @return The registered computation function, or an empty function if none
   * registered
   */
  static PropertyComputeFunction getPropertyComputation(PropertyType type) {
    auto it = s_propertyComputations.find(type);
    if (it != s_propertyComputations.end()) {
      return it->second;
    }
    // Return an empty function that does nothing
    return [](Cell &, const std::array<Cell *, 4> *) {};
  }

private:
  Grid *m_grid; ///<    // Reference to the grid this cell belongs to (needed
                ///<    for boundary conditions)
  // Boundary condition for this cell (if it's a boundary)
  std::shared_ptr<BoundaryClass> m_boundaryCondition;

  CellType m_type = CellType::FLUID; ///< Type of the cell

  double m_temperature = 293.15;    ///< Temperature at cell center (K)
  double m_pressure = 101325.0;     ///< Pressure at cell center (Pa)
  double m_density = 1.0;           ///< Density at cell center (kg/m³)
  std::array<Vertex, 4> m_vertices; ///< Vertices of the cell
  std::shared_ptr<Material> m_material =
      nullptr;            ///< Material properties of the cell
  bool m_isFixed = false; ///< Whether the cell has fixed values
  bool m_is_obstacle;
  bool m_is_boundary;
  // Cell center velocities (for Grid compatibility)
  double m_velocity_x = 0.0; ///< X-velocity at cell center
  double m_velocity_y = 0.0; ///< Y-velocity at cell center

  // Fixed properties for common numerical values (fast access)
  std::array<double, static_cast<size_t>(PropertyType::COUNT)>
      m_fixedProperties = {};

  // Boolean flags packed into bits (fastest access)
  uint64_t m_flags = 0;

  // Dynamic properties (flexible but slower access)
  std::unordered_map<std::string, double> m_dynamicProperties;

  // Hash function for dynamic properties
  static uint32_t hashName(const std::string &name);

  static 
  std::unordered_map<PropertyType, PropertyComputeFunction>
      s_propertyComputations;
};
