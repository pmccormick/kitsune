/**
 * ====================================================================
 * Cell Class - Optimized Implementation for Cache and SIMD
 * ====================================================================
 *
 * This header defines the optimized Cell class implementation that uses
 * flattened data structures for better cache locality and vectorization.
 */

#pragma once

#include "Units.h"
#include <array>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declarations
class Material;
class Grid;
class BoundaryClass;

/**
 * @class Cell
 * @brief Represents a single cell in the computational grid for CFD simulation
 *
 * This class is optimized for cache locality, SIMD operations, and future GPU
 * compatibility by using flattened, aligned data structures.
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
    VORTICITY = 0,           ///< Curl of velocity field (rotation)
    STREAM_FUNCTION = 1,     ///< Stream function value (for visualization)
    KINETIC_ENERGY = 2,      ///< Kinetic energy per unit mass
    DIVERGENCE = 3,          ///< Velocity divergence
    PRESSURE_CORRECTION = 4, ///< Pressure correction term
    HEAT_FLUX_X = 5,         ///< Heat flux in x-direction
    HEAT_FLUX_Y = 6,         ///< Heat flux in y-direction
    SHEAR_STRESS = 7,        ///< Local shear stress
    WALL_DISTANCE = 8,       ///< Distance to nearest wall
    COUNT                    ///< Keep last - used for array sizing
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
    IS_BOUNDARY = 0x00000010, ///< Cell is a boundary cell
    IS_OBSTACLE = 0x00000020, ///< Cell is an obstacle
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

  // Constructors and Destructor
  Cell(Grid *grid = nullptr);
  explicit Cell(CellType type, Grid *grid = nullptr);
  ~Cell() = default;

  // Boundary Condition Methods
  void setBoundaryCondition(std::shared_ptr<BoundaryClass> boundary);
  std::shared_ptr<BoundaryClass> getBoundaryCondition() const;
  bool hasBoundaryCondition() const;
  bool hasTemperature() const {
    // Always return true since temperature is always available
    return true;
  }
  bool hasDensity() const {
    // Always return true since density is always available
    return true;
  }

  // Core Properties
  Grid *getGrid() const;
  CellType getType() const;
  void setType(CellType type);
  double getTemperature() const;
  void setTemperature(double temperature);
  double getPressure() const;
  void setPressure(double pressure);
  double getDensity() const;
  void setDensity(double density);
  std::shared_ptr<Material> getMaterial() const;
  void setMaterial(std::shared_ptr<Material> material);
  void setMaterial(Material *material);
  bool isFixed() const;
  void setFixed(bool fixed);

  // Temperature with unit conversion
  void setTemperatureWithUnits(double temperature, const std::string &unit);
  double getTemperatureWithUnits(const std::string &unit) const;

  // Pressure with unit conversion
  void setPressureWithUnits(double pressure, const std::string &unit);
  double getPressureWithUnits(const std::string &unit) const;

  // Density with unit conversion
  void setDensityWithUnits(double density, const std::string &unit);
  double getDensityWithUnits(const std::string &unit) const;

  // Velocity Methods
  std::pair<double, double> getVertexVelocity(VertexPosition position) const;
  void setVertexVelocity(VertexPosition position, double vx, double vy);
  Vertex &getVertex(VertexPosition position);
  const Vertex &getVertex(VertexPosition position) const;

  // Velocity with unit conversion
  void setVelocityWithUnits(double vx, double vy, const std::string &unit);
  double getVelocityUWithUnits(const std::string &unit) const;
  double getVelocityVWithUnits(const std::string &unit) const;
  void setVertexVelocityWithUnits(VertexPosition position, double vx, double vy,
                                  const std::string &unit);
  std::pair<double, double>
  getVertexVelocityWithUnits(VertexPosition position,
                             const std::string &unit) const;

  // Property Methods
  void setProperty(PropertyType type, double value);
  double getProperty(PropertyType type) const;

  // Convenience methods for specific properties
  double getVorticity() const;
  void setVorticity(double value);
  double getStreamFunction() const;
  void setStreamFunction(double value);
  double getKineticEnergy() const;
  void setKineticEnergy(double value);
  double getDivergence() const;
  void setDivergence(double value);
  double getPressureCorrection() const;
  void setPressureCorrection(double value);
  double getHeatFluxX() const;
  void setHeatFluxX(double value);
  double getHeatFluxY() const;
  void setHeatFluxY(double value);
  double getShearStress() const;
  void setShearStress(double value);
  double getWallDistance() const;
  void setWallDistance(double value);

  // Flag Methods
  void setFlag(CellFlag flag, bool value);
  bool getFlag(CellFlag flag) const;

  // Dynamic Property Methods
  void setDynamicProperty(const std::string &name, double value);
  double getDynamicProperty(const std::string &name,
                            double defaultValue = 0.0) const;

  // State Management
  void reset();

  // Grid Compatibility Methods
  void setBoundary(bool isBoundary);
  bool isBoundary() const;
  void setObstacle(bool isObstacle);
  bool isObstacle() const;
  double getVelocityU() const;
  double getVelocityV() const;
  void setVelocityU(double vx);
  void setVelocityV(double vy);

  // Derived Properties Computation
  using PropertyComputeFunction =
      std::function<void(Cell &, const std::array<Cell *, 4> *)>;
  static void registerPropertyComputation(PropertyType type,
                                          PropertyComputeFunction computeFunc);
  void
  computeDerivedProperties(const std::array<Cell *, 4> *neighbors = nullptr);
  static PropertyComputeFunction getPropertyComputation(PropertyType type);

  // Output and Serialization
  void print(std::ostream &os, int verbosity = 1) const;
  std::string toString(int verbosity = 1) const;
  std::string serialize() const;
  bool deserialize(const std::string &data);
  std::string toSVG(double scale = 10.0, bool showVelocity = true) const;

  // Static Helpers
  static std::string cellTypeToString(CellType type);
  static std::string cellFlagToString(CellFlag flag);

  // Friend for efficient implementation
  friend class Grid;

private:
  // ====================================================================
  // OPTIMIZED DATA LAYOUT FOR BETTER CACHE PERFORMANCE
  // ====================================================================

  // Integer and boolean values packed together for better cache locality
  struct CellState {
    CellType type = CellType::FLUID;
    bool isFixed = false;
    bool isObstacle = false;
    bool isBoundary = false;
    uint64_t flags = 0;
  } m_state;

  // Basic physical properties grouped together (frequently accessed together)
  alignas(64) struct PhysicalProperties {
    double temperature = 293.15; // K
    double pressure = 101325.0;  // Pa
    double density = 1.0;        // kg/m³
    double velocity_u = 0.0;     // m/s
    double velocity_v = 0.0;     // m/s
  } m_physics;

  // Vertex velocities in a flat array for better SIMD
  // Layout: [nw.vx, nw.vy, ne.vx, ne.vy, se.vx, se.vy, sw.vx, sw.vy]
  alignas(64) std::array<double, 8> m_vertexVelocities = {0};

  // Fixed properties in contiguous memory
  alignas(64) std::array<
      double, static_cast<size_t>(PropertyType::COUNT)> m_fixedProperties = {};

  // References to external objects
  Grid *m_grid = nullptr;
  std::shared_ptr<Material> m_material = nullptr;
  std::shared_ptr<BoundaryClass> m_boundaryCondition = nullptr;

  // Dynamic properties - could be further optimized if needed
  std::unordered_map<std::string, double> m_dynamicProperties;

  // Static storage for property computation functions
  static std::unordered_map<PropertyType, PropertyComputeFunction>
      s_propertyComputations;

  // Helper methods
  static uint32_t hashName(const std::string &name);
  double getVertexVelocityComponent(VertexPosition position, bool isY) const;
  void setVertexVelocityComponent(VertexPosition position, bool isY,
                                  double value);
};
