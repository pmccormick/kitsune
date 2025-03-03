#pragma once

#include <array>
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>

// Forward declarations
class Material;

/**
 * @class Cell
 * @brief Represents a single cell in the computational grid for CFD simulation
 * 
 * This class stores physical properties like temperature and pressure at the cell center,
 * while velocities are stored at the cell vertices (staggered grid approach).
 * It uses a hybrid approach for property storage to maximize performance.
 */
class Cell {
public:
  /**
   * @enum CellType
   * @brief Defines the possible types of cells in the simulation
   */
  enum class CellType {
    FLUID,       ///< Regular fluid cell
    SOLID,       ///< Solid obstacle cell (e.g., cylinder)
    BOUNDARY     ///< Boundary cell
  };
    
  /**
   * @enum VertexPosition
   * @brief Defines the relative positions of cell vertices
   * 
   * For a 2D rectangular cell:
   * NW---N---NE
   * |         |
   * W    C    E
   * |         |
   * SW---S---SE
   * 
   * where C is the cell center
   */
  enum class VertexPosition {
    NORTHWEST = 0,  // Top left
    NORTHEAST = 1,  // Top right
    SOUTHEAST = 2,  // Bottom right
    SOUTHWEST = 3   // Bottom left
  };
    
  /**
   * @enum PropertyType
   * @brief Defines the common numerical properties that are frequently accessed
   */
  enum class PropertyType {
    VORTICITY = 0,         ///< Curl of velocity field (rotation)
    STREAM_FUNCTION = 1,   ///< Stream function value (for visualization)
    KINETIC_ENERGY = 2,    ///< Kinetic energy per unit mass
    DIVERGENCE = 3,        ///< Velocity divergence (should be near zero for incompressible flow)
    PRESSURE_CORRECTION = 4, ///< Pressure correction term in SIMPLE/PISO algorithms
    HEAT_FLUX_X = 5,       ///< Heat flux in x-direction
    HEAT_FLUX_Y = 6,       ///< Heat flux in y-direction
    SHEAR_STRESS = 7,      ///< Local shear stress
    WALL_DISTANCE = 8,     ///< Distance to nearest wall (for turbulence models)
    COUNT                  ///< Keep last - used for array sizing
  };
    
  /**
   * @enum CellFlag
   * @brief Boolean flags for various cell states and behaviors
   */
  enum class CellFlag {
    // Boundary and type flags
    IS_INLET = 0,           ///< Cell is part of an inlet boundary
    IS_OUTLET = 1,          ///< Cell is part of an outlet boundary
    IS_WALL = 2,            ///< Cell is part of a wall boundary
    IS_SYMMETRY = 3,        ///< Cell is part of a symmetry boundary
        
    // Computational flags
    NEEDS_UPDATE = 4,       ///< Cell needs recalculation in current iteration
    ///< Used for adaptive mesh refinement or selective updates
    ///< to avoid unnecessary recalculation of stable regions
                                
    HAS_CONVERGED = 5,      ///< Local convergence flag for iterative solvers
    ///< When most cells have converged, the simulation can advance
                                
    SKIP_IN_PRESSURE_SOLVE = 6, ///< Skip this cell in pressure solution
    ///< Used for solid cells or special boundaries
                                    
    IS_IN_WAKE = 7,         ///< Cell is in wake region (for adaptive refinement)
    ///< Can be used to apply different numerical schemes
                                
    IS_IN_BOUNDARY_LAYER = 8, ///< Cell is within the boundary layer
    ///< Important for turbulence models and wall functions
                                 
    IS_IN_RECIRCULATION = 9, ///< Cell is in a recirculation zone
    ///< Useful for visualization and adaptive time-stepping
                                 
    HAS_EXTREME_GRADIENT = 10, ///< Cell has a steep gradient in a key property
    ///< Might need special numerical treatment
                                  
    MARKED_FOR_REFINEMENT = 11, ///< Cell is marked for mesh refinement
    ///< For adaptive mesh refinement strategies
                                    
    IS_NEWLY_CREATED = 12,     ///< Cell was recently created by mesh refinement
    ///< Might need special interpolation for initial values
                                   
    IS_TRANSITIONING = 13,     ///< Cell is in flow transition region
    ///< For turbulence modeling
                                   
    CONTAINS_INTERFACE = 14,   ///< Cell contains fluid interface
    ///< For multiphase flows
                                   
    COUNT                      ///< Keep last - used for bit field sizing
  };
    
  /**
   * @struct Vertex
   * @brief Data structure for cell vertex information
   */
  struct Vertex {
    double vx = 0.0;        ///< x-component of velocity
    double vy = 0.0;        ///< y-component of velocity
  };
    
  /**
   * @brief Default constructor
   */
  Cell();
    
  /**
   * @brief Parameterized constructor
   * @param type The type of cell (fluid, solid, boundary)
   */
  explicit Cell(CellType type);
    
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
   * @brief Get reference to a vertex
   * @param position The vertex position
   * @return Reference to the vertex
   */
  Vertex& getVertex(VertexPosition position);
    
  /**
   * @brief Get const reference to a vertex
   * @param position The vertex position
   * @return Const reference to the vertex
   */
  const Vertex& getVertex(VertexPosition position) const;
    
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
  double getProperty(PropertyType type, double defaultValue = 0.0) const;
    
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
  void setDynamicProperty(const std::string& name, double value);
    
  /**
   * @brief Get a dynamic property value (slower access)
   * @param name The property name
   * @param defaultValue Default value if not found
   * @return The property value
   */
  double getDynamicProperty(const std::string& name, double defaultValue = 0.0) const;
    
  /**
   * @brief Reset the cell to default values
   */
  void reset();
    
private:
  CellType m_type = CellType::FLUID;                 ///< Type of the cell
  double m_temperature = 293.15;                     ///< Temperature at cell center (K)
  double m_pressure = 101325.0;                      ///< Pressure at cell center (Pa)
  double m_density = 1.0;                            ///< Density at cell center (kg/m³)
  std::array<Vertex, 4> m_vertices;                  ///< Vertices of the cell
  std::shared_ptr<Material> m_material = nullptr;    ///< Material properties of the cell
  bool m_isFixed = false;                            ///< Whether the cell has fixed values
    
  // Fixed properties for common numerical values (fast access)
  std::array<double, static_cast<size_t>(PropertyType::COUNT)> m_fixedProperties = {};
    
  // Boolean flags packed into bits (fastest access)
  uint32_t m_flags = 0;
    
  // Dynamic properties (flexible but slower access)
  std::unordered_map<std::string, double> m_dynamicProperties;
    
  // Hash function for dynamic properties
  static uint32_t hashName(const std::string& name);
};

