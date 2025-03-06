/**
 * @file Cell.h
 * @brief Cell class for CFD simulations with Field-based property storage.
 *
 * ================================================================================
 * Design Considerations and Future Directions:
 * ================================================================================
 * This Cell class now serves as a thin access layer over the Field objects
 * owned by the Mesh. Key design points include:
 *
 *   • **Facade Pattern with Field Backend:**
 *     Instead of directly storing physical quantities, this Cell acts as a
 * facade that provides access to data stored in contiguous Field arrays. This
 * allows for cache-friendly operations while maintaining an intuitive
 * cell-based API.
 *
 *   • **Index-Based Access:**
 *     Each Cell stores its (i,j) indices in the grid and uses these to access
 * the appropriate elements in the corresponding Fields.
 *
 *   • **Full Simulation Interface Preservation:**
 *     The Cell class still supports all the same operations (property access,
 * type management, flags, etc.) but now delegates storage to Fields.
 *
 *   • **Material Integration and Mixing:**
 *     A cell's material is now stored in a dedicated material Field owned by
 * the Mesh, ensuring contiguous memory layout for all data.
 *
 *   • **Performance Benefits:**
 *     This design enables:
 *     - Better cache utilization through contiguous memory access
 *     - Potential for vectorized operations on entire Fields
 *     - Reduced memory fragmentation and improved allocation patterns
 *     - Easier integration with GPU acceleration
 *
 * ================================================================================
 * Role in Computational Science:
 * ================================================================================
 * This refactored Cell design bridges the gap between intuitive object-oriented
 * APIs and high-performance data-oriented implementations. By providing a
 * familiar cell-centric interface while leveraging modern hardware-friendly
 * data layouts, this design:
 *
 *   - Maintains compatibility with existing CFD algorithms and boundary
 * conditions
 *   - Improves performance through better memory access patterns
 *   - Provides a path to parallelization (both CPU SIMD and GPU)
 *   - Preserves the physical correctness and unit safety of the original design
 *
 * ================================================================================
 */

#ifndef CELL_H
#define CELL_H

#include "BoundaryClass.h"
#include "Field.h"
#include "Material.h"
#include "Units.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>

// Forward declaration
class Mesh;

//------------------------------------------------------------------------------
// Enumerations and supporting types
//------------------------------------------------------------------------------
class Cell {
public:
  // Cell types.
  enum class CellType { FLUID = 0, SOLID, BOUNDARY };

  // Positions for vertex velocity access.
  enum class VertexPosition { NORTHWEST = 0, NORTHEAST, SOUTHEAST, SOUTHWEST };

  // Fixed properties indices.
  enum class PropertyType {
    VORTICITY = 0,
    STREAM_FUNCTION,
    KINETIC_ENERGY,
    DIVERGENCE,
    PRESSURE_CORRECTION,
    HEAT_FLUX_X,
    HEAT_FLUX_Y,
    SHEAR_STRESS,
    WALL_DISTANCE,
    COUNT
  };

  // Flag definitions (using bitfields).
  enum class CellFlag : uint32_t {
    IS_INLET = 0x00000001,
    IS_OUTLET = 0x00000002,
    IS_WALL = 0x00000004,
    IS_SYMMETRY = 0x00000008,
    IS_BOUNDARY = 0x00000010,
    IS_OBSTACLE = 0x00000020,
    COUNT = 0x00000040 // not used as a flag
  };

  // Vertex structure for storing vertex velocities.
  struct Vertex {
    double vx = 0.0;
    double vy = 0.0;
  };

  // Constructor for a cell at the specified grid location
  Cell(Mesh *mesh, size_t i, size_t j);

  // Default constructor for creating temporary cells
  Cell();

  // Constructors with type specification.
  Cell(Mesh *mesh, size_t i, size_t j, CellType type);

  // Destructor.
  ~Cell() = default;

  // --- Type and Flag Operations ---
  CellType getType() const;
  void setType(CellType type);

  bool isFixed() const;
  void setFixed(bool fixed);

  bool isBoundary() const;
  void setBoundary(bool isBoundary);

  bool isObstacle() const;
  void setObstacle(bool isObstacle);

  bool getFlag(CellFlag flag) const;
  void setFlag(CellFlag flag, bool value);

  // --- Cell-Centered Physical Property Accessors (SI Units) ---
  double getTemperature() const;
  void setTemperature(double temperature);
  bool hasBoundaryCondition() const;
  bool hasTemperature() const { return true; }
  bool hasDensity() const { return true; }

  double getPressure() const;
  void setPressure(double pressure);

  double getDensity() const;
  void setDensity(double density);

  double getVelocityU() const; // x-velocity
  void setVelocityU(double vx);

  double getVelocityV() const; // y-velocity
  void setVelocityV(double vy);

  // --- Unit Conversion Functions ---
  // Temperature conversion.
  void setTemperatureWithUnits(double temperature, const std::string &unit);
  double getTemperatureWithUnits(const std::string &unit) const;

  // Pressure conversion.
  void setPressureWithUnits(double pressure, const std::string &unit);
  double getPressureWithUnits(const std::string &unit) const;

  // Density conversion.
  void setDensityWithUnits(double density, const std::string &unit);
  double getDensityWithUnits(const std::string &unit) const;

  // Velocity conversion.
  void setVelocityWithUnits(double vx, double vy, const std::string &unit);
  double getVelocityUWithUnits(const std::string &unit) const;
  double getVelocityVWithUnits(const std::string &unit) const;

  // Vertex velocity access (for cell corners).
  std::pair<double, double> getVertexVelocity(VertexPosition pos) const;
  void setVertexVelocity(VertexPosition pos, double vx, double vy);
  void setVertexVelocityWithUnits(VertexPosition pos, double vx, double vy,
                                  const std::string &unit);
  std::pair<double, double>
  getVertexVelocityWithUnits(VertexPosition pos, const std::string &unit) const;

  // Access to the parent Mesh
  Mesh *getGrid() const { return m_mesh; }

  // Boundary condition access
  std::shared_ptr<BoundaryClass> getBoundaryCondition() const;
  void setBoundaryCondition(std::shared_ptr<BoundaryClass> bc);

  // --- Material Integration ---
  std::shared_ptr<Material> getMaterial() const;
  void setMaterial(std::shared_ptr<Material> material);
  /**
   * @brief Mixes the cell's current material with another using the Material
   * class's mixing function.
   * @param other Shared pointer to the other material.
   * @param mixFraction Fraction of the other material.
   * @param mixingRule Mixing rule string (default uses Material's default).
   */
  void mixMaterial(const std::shared_ptr<Material> &other, double mixFraction,
                   const std::string &mixingRule = "default");

  double getEffectiveDensity() const;

  // --- Dynamic and Fixed Property Access ---
  void setProperty(PropertyType type, double value);
  double getProperty(PropertyType type) const;

  // Convenience methods.
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

  void setDynamicProperty(const std::string &name, double value);
  double getDynamicProperty(const std::string &name,
                            double defaultValue = 0.0) const;

  // --- Property Computation Registration ---
  using PropertyComputeFunction =
      std::function<void(Cell &, const std::array<Cell *, 4> *)>;

  // Register a computation function for a property
  static void registerPropertyComputation(PropertyType type,
                                          PropertyComputeFunction func);

  // Get a computation function for a property
  static PropertyComputeFunction getPropertyComputation(PropertyType type);

  // Compute all derived properties for this cell
  void
  computeDerivedProperties(const std::array<Cell *, 4> *neighbors = nullptr);

  // --- State Management ---
  void reset();

  // --- Serialization ---
  std::string serialize() const;
  bool deserialize(const std::string &data);

  // --- Diagnostic Output ---
  void print(std::ostream &os, int verbosity = 1) const;
  std::string toString(int verbosity = 1) const;
  std::string toSVG(double scale = 10.0, bool showVelocity = true) const;

  // --- Static Helper Methods ---
  static std::string cellTypeToString(CellType type);
  static std::string cellFlagToString(CellFlag flag);

  // --- Index Access ---
  // Get grid indices
  size_t getI() const { return m_i; }
  size_t getJ() const { return m_j; }

private:
  // Reference to the parent mesh that owns the fields
  Mesh *m_mesh;

  // Grid indices
  size_t m_i;
  size_t m_j;

  // Cached dynamic properties for non-field properties
  std::unordered_map<std::string, double> m_dynamicProperties;

  // Static registry for property computation functions
  static std::unordered_map<PropertyType, PropertyComputeFunction>
      s_propertyComputeFunctions;
};

#endif // CELL_H