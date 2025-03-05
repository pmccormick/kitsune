#include "Cell.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>

// Forward declaration if needed
#include "Material.h"
#include "BoundaryClass.h"
#include "BoundaryFactory.h"

// Static hash function for property names
uint32_t Cell::hashName(const std::string &name) {
  // Simple hash function (could use std::hash)
  uint32_t hash = 0;
  for (char c : name) {
    hash = hash * 31 + c;
  }
  return hash;
}

// Static initialization of the property computation map
std::unordered_map<Cell::PropertyType, Cell::PropertyComputeFunction>
    Cell::s_propertyComputations;

// ====================================================================
// Constructors
// ====================================================================

Cell::Cell(Grid *grid) : m_grid(grid) {
  // Default initialization is handled by the inline initializers in the header
}

Cell::Cell(CellType type, Grid *grid) : m_grid(grid) {
  // Set type and update related flags
  setType(type);
}

// ====================================================================
// Simple Accessors and Mutators for Basic Properties
// ====================================================================

// Boundary condition methods
void Cell::setBoundaryCondition(std::shared_ptr<BoundaryClass> boundary) {
  m_boundaryCondition = boundary;
}

std::shared_ptr<BoundaryClass> Cell::getBoundaryCondition() const {
  return m_boundaryCondition;
}

bool Cell::hasBoundaryCondition() const {
  return m_boundaryCondition != nullptr;
}

// Grid access
Grid *Cell::getGrid() const { return m_grid; }

// Type methods
Cell::CellType Cell::getType() const { return m_state.type; }

void Cell::setType(CellType type) {
  m_state.type = type;

  // Handle fixed state based on type
  setFixed(type == CellType::BOUNDARY || type == CellType::SOLID);

  // Update boundary/obstacle flags based on type
  if (type == CellType::BOUNDARY) {
    m_state.isBoundary = true;
    m_state.isObstacle = false;
    m_state.flags |= static_cast<uint64_t>(CellFlag::IS_BOUNDARY);
    m_state.flags &= ~static_cast<uint64_t>(CellFlag::IS_OBSTACLE);
  } else if (type == CellType::SOLID) {
    m_state.isBoundary = false;
    m_state.isObstacle = true;
    m_state.flags |= static_cast<uint64_t>(CellFlag::IS_OBSTACLE);
    m_state.flags &= ~static_cast<uint64_t>(CellFlag::IS_BOUNDARY);
  } else { // FLUID
    m_state.isBoundary = false;
    m_state.isObstacle = false;
    m_state.flags &= ~static_cast<uint64_t>(CellFlag::IS_BOUNDARY);
    m_state.flags &= ~static_cast<uint64_t>(CellFlag::IS_OBSTACLE);
  }
}

// Temperature methods
double Cell::getTemperature() const { return m_physics.temperature; }

void Cell::setTemperature(double temperature) {
  m_physics.temperature = Units::enforceValidTemperature(temperature);
}

void Cell::setTemperatureWithUnits(double temperature,
                                   const std::string &unit) {
  double kelvin = Units::convert(temperature, unit, "K");
  setTemperature(kelvin);
}

double Cell::getTemperatureWithUnits(const std::string &unit) const {
  return Units::convert(getTemperature(), "K", unit);
}

// Pressure methods
double Cell::getPressure() const { return m_physics.pressure; }

void Cell::setPressure(double pressure) {
  m_physics.pressure = Units::enforceValidPressure(pressure);
}

void Cell::setPressureWithUnits(double pressure, const std::string &unit) {
  double pascal = Units::convert(pressure, unit, "Pa");
  setPressure(pascal);
}

double Cell::getPressureWithUnits(const std::string &unit) const {
  return Units::convert(getPressure(), "Pa", unit);
}

// Density methods
double Cell::getDensity() const { return m_physics.density; }

void Cell::setDensity(double density) {
  m_physics.density = Units::enforceValidDensity(density);
}

void Cell::setDensityWithUnits(double density, const std::string &unit) {
  double kgm3 = Units::convert(density, unit, "kg/m³");
  setDensity(kgm3);
}

double Cell::getDensityWithUnits(const std::string &unit) const {
  return Units::convert(getDensity(), "kg/m³", unit);
}

// Material methods
std::shared_ptr<Material> Cell::getMaterial() const { return m_material; }

void Cell::setMaterial(std::shared_ptr<Material> material) {
  m_material = material;
}

void Cell::setMaterial(Material *material) {
  if (material != nullptr) {
    // Create a shared_ptr that doesn't delete the Material (Grid owns it)
    m_material = std::shared_ptr<Material>(material, [](Material *) {});
  } else {
    m_material = nullptr;
  }
}

// Fixed state methods
bool Cell::isFixed() const { return m_state.isFixed; }

void Cell::setFixed(bool fixed) { m_state.isFixed = fixed; }

// ====================================================================
// Velocity Methods
// ====================================================================

double Cell::getVelocityU() const { return m_physics.velocity_u; }

double Cell::getVelocityV() const { return m_physics.velocity_v; }

void Cell::setVelocityU(double vu) { m_physics.velocity_u = vu; }

void Cell::setVelocityV(double vv) { m_physics.velocity_v = vv; }

void Cell::setVelocityWithUnits(double vx, double vy, const std::string &unit) {
  double vx_mps = Units::convert(vx, unit, "m/s");
  double vy_mps = Units::convert(vy, unit, "m/s");
  setVelocityU(vx_mps);
  setVelocityV(vy_mps);
}

double Cell::getVelocityUWithUnits(const std::string &unit) const {
  return Units::convert(getVelocityU(), "m/s", unit);
}

double Cell::getVelocityVWithUnits(const std::string &unit) const {
  return Units::convert(getVelocityV(), "m/s", unit);
}

// ====================================================================
// Flag Methods
// ====================================================================

void Cell::setFlag(CellFlag flag, bool value) {
  // Skip COUNT which isn't a real flag
  if (flag == CellFlag::COUNT)
    return;

  if (value) {
    // Handle mutually exclusive flags
    if (flag == CellFlag::IS_INLET) {
      // Clear outlet flag if setting inlet
      m_state.flags &= ~static_cast<uint64_t>(CellFlag::IS_OUTLET);
    } else if (flag == CellFlag::IS_OUTLET) {
      // Clear inlet flag if setting outlet
      m_state.flags &= ~static_cast<uint64_t>(CellFlag::IS_INLET);
    }

    // Set the requested flag
    m_state.flags |= static_cast<uint64_t>(flag);
  } else {
    // Clear the requested flag
    m_state.flags &= ~static_cast<uint64_t>(flag);
  }
}

bool Cell::getFlag(CellFlag flag) const {
  // Skip COUNT which isn't a real flag
  if (flag == CellFlag::COUNT)
    return false;

  return (m_state.flags & static_cast<uint64_t>(flag)) != 0;
}

// ====================================================================
// Boundary/Obstacle Methods
// ====================================================================

void Cell::setBoundary(bool isBoundary) {
  m_state.isBoundary = isBoundary;
  setFlag(CellFlag::IS_BOUNDARY, isBoundary);
  if (isBoundary) {
    setType(CellType::BOUNDARY);
  }
}

bool Cell::isBoundary() const { return m_state.isBoundary; }

void Cell::setObstacle(bool isObstacle) {
  m_state.isObstacle = isObstacle;
  setFlag(CellFlag::IS_OBSTACLE, isObstacle);
  if (isObstacle) {
    setType(CellType::SOLID);
  }
}

bool Cell::isObstacle() const { return m_state.isObstacle; }

// ====================================================================
// Simple String Conversion Methods
// ====================================================================

std::string Cell::cellTypeToString(CellType type) {
  switch (type) {
  case CellType::FLUID:
    return "FLUID";
  case CellType::SOLID:
    return "SOLID";
  case CellType::BOUNDARY:
    return "BOUNDARY";
  default:
    return "UNKNOWN";
  }
}

std::string Cell::cellFlagToString(CellFlag flag) {
  switch (flag) {
  case CellFlag::IS_INLET:
    return "INLET";
  case CellFlag::IS_OUTLET:
    return "OUTLET";
  case CellFlag::IS_WALL:
    return "WALL";
  case CellFlag::IS_SYMMETRY:
    return "SYMMETRY";
  case CellFlag::IS_BOUNDARY:
    return "BOUNDARY";
  case CellFlag::IS_OBSTACLE:
    return "OBSTACLE";
  case CellFlag::COUNT:
    return "COUNT";
  default:
    return "UNKNOWN";
  }
}

// ====================================================================
// Vertex-Related Methods for Cell Class
// ====================================================================

// Helper method to calculate the index in the flattened vertex array
size_t getVertexIndex(Cell::VertexPosition position, bool isY) {
  // Each vertex has 2 components (vx, vy)
  // The array is [nw.vx, nw.vy, ne.vx, ne.vy, se.vx, se.vy, sw.vx, sw.vy]
  size_t vertexBaseIndex = static_cast<size_t>(position) * 2;
  return vertexBaseIndex + (isY ? 1 : 0);
}

// Helper method to get vertex velocity component (either vx or vy)
double Cell::getVertexVelocityComponent(VertexPosition position,
                                        bool isY) const {
  size_t index = getVertexIndex(position, isY);
  return m_vertexVelocities[index];
}

// Helper method to set vertex velocity component (either vx or vy)
void Cell::setVertexVelocityComponent(VertexPosition position, bool isY,
                                      double value) {
  size_t index = getVertexIndex(position, isY);
  m_vertexVelocities[index] = value;
}

// Get both velocity components for a vertex
std::pair<double, double>
Cell::getVertexVelocity(VertexPosition position) const {
  double vx = getVertexVelocityComponent(position, false);
  double vy = getVertexVelocityComponent(position, true);
  return {vx, vy};
}

// Set both velocity components for a vertex
void Cell::setVertexVelocity(VertexPosition position, double vx, double vy) {
  setVertexVelocityComponent(position, false, vx);
  setVertexVelocityComponent(position, true, vy);
}

// Get vertex with unit conversion
std::pair<double, double>
Cell::getVertexVelocityWithUnits(VertexPosition position,
                                 const std::string &unit) const {
  auto [vx_mps, vy_mps] = getVertexVelocity(position);
  return {Units::convert(vx_mps, "m/s", unit),
          Units::convert(vy_mps, "m/s", unit)};
}

// Set vertex with unit conversion
void Cell::setVertexVelocityWithUnits(VertexPosition position, double vx,
                                      double vy, const std::string &unit) {
  double vx_mps = Units::convert(vx, unit, "m/s");
  double vy_mps = Units::convert(vy, unit, "m/s");
  setVertexVelocity(position, vx_mps, vy_mps);
}

// Get reference to vertex structure
// This requires creating a temporary Vertex object since we're storing
// components in a flat array
Cell::Vertex &Cell::getVertex(VertexPosition position) {
  static thread_local Vertex tempVertex;
  auto [vx, vy] = getVertexVelocity(position);
  tempVertex.vx = vx;
  tempVertex.vy = vy;

  // Warning: This returns a reference to a static temporary
  // Any modifications to it won't affect the actual stored values
  // The calling code should use setVertexVelocity() to make changes
  return tempVertex;
}

// Get const reference to vertex structure
const Cell::Vertex &Cell::getVertex(VertexPosition position) const {
  static thread_local Vertex tempVertex;
  auto [vx, vy] = getVertexVelocity(position);
  tempVertex.vx = vx;
  tempVertex.vy = vy;

  // Same warning as above
  return tempVertex;
}

// Completion of print method
void Cell::print(std::ostream &os, int verbosity) const {
  os << "Cell [Type: " << cellTypeToString(m_state.type) << "]" << std::endl;

  if (verbosity >= 1) {
    // Basic properties
    os << "  Temperature: " << m_physics.temperature << " K ("
       << getTemperatureWithUnits("C") << " °C)" << std::endl;
    os << "  Pressure: " << m_physics.pressure << " Pa ("
       << getPressureWithUnits("bar") << " bar)" << std::endl;
    os << "  Density: " << m_physics.density << " kg/m³" << std::endl;
    os << "  Velocity: [" << m_physics.velocity_u << ", "
       << m_physics.velocity_v << "] m/s" << std::endl;

    // State flags
    os << "  Fixed: " << (isFixed() ? "Yes" : "No") << std::endl;
    os << "  Is Boundary: " << (isBoundary() ? "Yes" : "No") << std::endl;
    os << "  Is Obstacle: " << (isObstacle() ? "Yes" : "No") << std::endl;

    // Material
    os << "  Material: " << (m_material ? m_material->getName() : "None")
       << std::endl;
  }

  if (verbosity >= 2) {
    // Detailed - include property values
    os << "  Properties:" << std::endl;
    os << "    Vorticity: " << getVorticity() << " 1/s" << std::endl;
    os << "    Kinetic Energy: " << getKineticEnergy() << " J/kg" << std::endl;
    os << "    Stream Function: " << getStreamFunction() << " m²/s"
       << std::endl;
    os << "    Divergence: " << getDivergence() << " 1/s" << std::endl;

    // Flags
    os << "  Flags:" << std::endl;
    for (int i = 0; i < static_cast<int>(CellFlag::COUNT); i++) {
      CellFlag flag = static_cast<CellFlag>(1 << i);
      if (getFlag(flag)) {
        os << "    " << cellFlagToString(flag) << std::endl;
      }
    }

    // Vertex velocities
    os << "  Vertex Velocities:" << std::endl;
    for (int i = 0; i < 4; i++) {
      VertexPosition pos = static_cast<VertexPosition>(i);
      auto [vx, vy] = getVertexVelocity(pos);
      os << "    "
         << (pos == VertexPosition::NORTHWEST   ? "NW"
             : pos == VertexPosition::NORTHEAST ? "NE"
             : pos == VertexPosition::SOUTHEAST ? "SE"
                                                : "SW")
         << ": [" << vx << ", " << vy << "] m/s" << std::endl;
    }

    // Dynamic properties
    if (!m_dynamicProperties.empty()) {
      os << "  Dynamic Properties:" << std::endl;
      for (const auto &[name, value] : m_dynamicProperties) {
        os << "    " << name << ": " << value << std::endl;
      }
    }
  }
}

// String representation of the cell
std::string Cell::toString(int verbosity) const {
  std::ostringstream oss;
  print(oss, verbosity);
  return oss.str();
}

// Generate SVG representation of the cell
std::string Cell::toSVG(double scale, bool showVelocity) const {
  std::ostringstream svg;

  // SVG header
  svg << "<svg width=\"" << 12 * scale << "\" height=\"" << 12 * scale
      << "\" xmlns=\"http://www.w3.org/2000/svg\">\n";

  // Cell background based on type
  std::string fillColor;
  switch (m_state.type) {
  case CellType::FLUID:
    fillColor = "rgb(200,230,255)"; // Light blue for fluid
    break;
  case CellType::SOLID:
    fillColor = "rgb(150,150,150)"; // Gray for solid
    break;
  case CellType::BOUNDARY:
    // Color based on boundary type
    if (getFlag(CellFlag::IS_INLET)) {
      fillColor = "rgb(100,200,100)"; // Green for inlet
    } else if (getFlag(CellFlag::IS_OUTLET)) {
      fillColor = "rgb(200,100,100)"; // Red for outlet
    } else if (getFlag(CellFlag::IS_WALL)) {
      fillColor = "rgb(120,120,120)"; // Dark gray for wall
    } else if (getFlag(CellFlag::IS_SYMMETRY)) {
      fillColor = "rgb(255,220,100)"; // Yellow for symmetry
    } else {
      fillColor = "rgb(200,200,200)"; // Light gray for generic boundary
    }
    break;
  }

  // Draw cell rectangle
  svg << "  <rect x=\"" << scale << "\" y=\"" << scale << "\" width=\""
      << 10 * scale << "\" height=\"" << 10 * scale << "\" fill=\"" << fillColor
      << "\" stroke=\"black\" stroke-width=\"1\"/>\n";

  // Draw temperature indication (red-blue gradient)
  double normTemp = (m_physics.temperature - 273.15) /
                    100.0; // Normalize around 0°C = 273.15K
  normTemp = std::max(0.0, std::min(1.0, normTemp)); // Clamp to [0,1]
  int red = static_cast<int>(255 * normTemp);
  int blue = static_cast<int>(255 * (1.0 - normTemp));
  svg << "  <circle cx=\"" << 3 * scale << "\" cy=\"" << 3 * scale << "\" r=\""
      << scale << "\" fill=\"rgb(" << red << ",0," << blue << ")\"/>\n";

  // Draw pressure indication (size of circle)
  double normPressure = m_physics.pressure / 101325.0; // Normalize around 1 atm
  normPressure = std::max(0.2, std::min(1.5, normPressure)); // Clamp and scale
  svg << "  <circle cx=\"" << 8 * scale << "\" cy=\"" << 3 * scale << "\" r=\""
      << scale * normPressure << "\" fill=\"rgba(0,0,0,0.3)\"/>\n";

  if (showVelocity) {
    // Draw velocity vector at center
    double velMag = std::sqrt(m_physics.velocity_u * m_physics.velocity_u +
                              m_physics.velocity_v * m_physics.velocity_v);
    if (velMag > 1e-6) {             // Only if non-zero
      double velScale = 3.0 * scale; // Scale factor for velocity arrows
      double normVelX = m_physics.velocity_u / velMag;
      double normVelY = m_physics.velocity_v / velMag;

      // Arrow from center
      double centerX = 6 * scale;
      double centerY = 6 * scale;
      double endX = centerX + normVelX * velScale * std::min(velMag, 3.0);
      double endY = centerY + normVelY * velScale * std::min(velMag, 3.0);

      // Arrow shaft
      svg << "  <line x1=\"" << centerX << "\" y1=\"" << centerY << "\" x2=\""
          << endX << "\" y2=\"" << endY
          << "\" stroke=\"black\" stroke-width=\"2\"/>\n";

      // Arrow head
      double arrowSize = 0.5 * scale;
      double angle = std::atan2(endY - centerY, endX - centerX);
      double arrow1X = endX - arrowSize * std::cos(angle - 0.5);
      double arrow1Y = endY - arrowSize * std::sin(angle - 0.5);
      double arrow2X = endX - arrowSize * std::cos(angle + 0.5);
      double arrow2Y = endY - arrowSize * std::sin(angle + 0.5);

      svg << "  <polygon points=\"" << endX << "," << endY << " " << arrow1X
          << "," << arrow1Y << " " << arrow2X << "," << arrow2Y
          << "\" fill=\"black\"/>\n";
    }
  }

  // Draw a small indicator for each property type that has a non-zero value
  double propRadius = 0.5 * scale;
  double startY = 9 * scale;
  for (int i = 0; i < static_cast<int>(PropertyType::COUNT); i++) {
    PropertyType propType = static_cast<PropertyType>(i);
    if (std::abs(getProperty(propType)) > 1e-6) {
      // Position indicators along the bottom
      double posX = (2 + i) * scale;
      svg << "  <circle cx=\"" << posX << "\" cy=\"" << startY << "\" r=\""
          << propRadius << "\" fill=\"purple\"/>\n";
    }
  }

  // SVG footer
  svg << "</svg>\n";

  return svg.str();
}

// ====================================================================
// Property Methods
// ====================================================================

// Set a fixed property value
void Cell::setProperty(PropertyType type, double value) {
  m_fixedProperties[static_cast<size_t>(type)] = value;
}

// Get a fixed property value
double Cell::getProperty(PropertyType type) const {
  return m_fixedProperties[static_cast<size_t>(type)];
}

// Register a computation function for a property type
void Cell::registerPropertyComputation(PropertyType type,
                                       PropertyComputeFunction computeFunc) {
  s_propertyComputations[type] = computeFunc;
}

// Get the computation function for a property type
Cell::PropertyComputeFunction Cell::getPropertyComputation(PropertyType type) {
  auto it = s_propertyComputations.find(type);
  if (it != s_propertyComputations.end()) {
    return it->second;
  }
  // Return an empty function that does nothing
  return [](Cell &, const std::array<Cell *, 4> *) {};
}

// Compute derived properties
void Cell::computeDerivedProperties(const std::array<Cell *, 4> *neighbors) {
  // Apply registered computation functions
  for (const auto &[type, computeFunc] : s_propertyComputations) {
    computeFunc(*this, neighbors);
  }
}

// Convenience property getter and setter methods

// Vorticity - local rotation in the flow field (1/s)
double Cell::getVorticity() const {
  return getProperty(PropertyType::VORTICITY);
}

void Cell::setVorticity(double value) {
  setProperty(PropertyType::VORTICITY, value);
}

// Stream function for 2D flow visualization (m²/s)
double Cell::getStreamFunction() const {
  return getProperty(PropertyType::STREAM_FUNCTION);
}

void Cell::setStreamFunction(double value) {
  setProperty(PropertyType::STREAM_FUNCTION, value);
}

// Kinetic energy per unit mass (J/kg or m²/s²)
double Cell::getKineticEnergy() const {
  return getProperty(PropertyType::KINETIC_ENERGY);
}

void Cell::setKineticEnergy(double value) {
  setProperty(PropertyType::KINETIC_ENERGY, value);
}

// Velocity divergence (1/s) - should be ~0 for incompressible flow
double Cell::getDivergence() const {
  return getProperty(PropertyType::DIVERGENCE);
}

void Cell::setDivergence(double value) {
  setProperty(PropertyType::DIVERGENCE, value);
}

// Pressure correction term for SIMPLE/PISO algorithms (Pa)
double Cell::getPressureCorrection() const {
  return getProperty(PropertyType::PRESSURE_CORRECTION);
}

void Cell::setPressureCorrection(double value) {
  setProperty(PropertyType::PRESSURE_CORRECTION, value);
}

// Heat flux components (W/m²)
double Cell::getHeatFluxX() const {
  return getProperty(PropertyType::HEAT_FLUX_X);
}

void Cell::setHeatFluxX(double value) {
  setProperty(PropertyType::HEAT_FLUX_X, value);
}

double Cell::getHeatFluxY() const {
  return getProperty(PropertyType::HEAT_FLUX_Y);
}

void Cell::setHeatFluxY(double value) {
  setProperty(PropertyType::HEAT_FLUX_Y, value);
}

// Local shear stress (Pa)
double Cell::getShearStress() const {
  return getProperty(PropertyType::SHEAR_STRESS);
}

void Cell::setShearStress(double value) {
  setProperty(PropertyType::SHEAR_STRESS, value);
}

// Distance to nearest wall for turbulence models (m)
double Cell::getWallDistance() const {
  return getProperty(PropertyType::WALL_DISTANCE);
}

void Cell::setWallDistance(double value) {
  setProperty(PropertyType::WALL_DISTANCE, value);
}

// Dynamic property methods
void Cell::setDynamicProperty(const std::string &name, double value) {
  m_dynamicProperties[name] = value;
}

double Cell::getDynamicProperty(const std::string &name,
                                double defaultValue) const {
  auto it = m_dynamicProperties.find(name);
  return (it != m_dynamicProperties.end()) ? it->second : defaultValue;
}

// ====================================================================
// Reset and Serialization Methods
// ====================================================================

// Reset the cell to default values
void Cell::reset() {
  // Preserve type and flags related to the type
  CellType type = m_state.type;
  bool isBoundary = m_state.isBoundary;
  bool isObstacle = m_state.isObstacle;

  // Reset physical properties
  m_physics.temperature = 293.15;
  m_physics.pressure = 101325.0;
  m_physics.density = 1.0;
  m_physics.velocity_u = 0.0;
  m_physics.velocity_v = 0.0;

  // Reset vertex velocities
  m_vertexVelocities.fill(0.0);

  // Reset fixed properties
  m_fixedProperties.fill(0.0);

  // Clear dynamic properties
  m_dynamicProperties.clear();

  // Restore type and related flags
  m_state.type = type;
  m_state.isBoundary = isBoundary;
  m_state.isObstacle = isObstacle;

  // Reset flags to only include boundary/obstacle as needed
  m_state.flags = 0;
  if (isBoundary) {
    m_state.flags |= static_cast<uint64_t>(CellFlag::IS_BOUNDARY);
    m_state.flags |=
        static_cast<uint64_t>(CellFlag::IS_WALL); // Default boundary is a wall
  }
  if (isObstacle) {
    m_state.flags |= static_cast<uint64_t>(CellFlag::IS_OBSTACLE);
  }

  // Material is preserved
}

// Complete implementation of the serialize method with boundary support
std::string Cell::serialize() const {
  std::ostringstream oss;

  // Version identifier to support future changes
  oss << "CELL_V1\n";

  // Cell state
  oss << "TYPE=" << static_cast<int>(m_state.type) << "\n";
  oss << "FIXED=" << (m_state.isFixed ? 1 : 0) << "\n";
  oss << "IS_BOUNDARY=" << (m_state.isBoundary ? 1 : 0) << "\n";
  oss << "IS_OBSTACLE=" << (m_state.isObstacle ? 1 : 0) << "\n";
  oss << "FLAGS=" << m_state.flags << "\n";

  // Physical properties
  oss << "TEMP=" << m_physics.temperature << "\n";
  oss << "PRES=" << m_physics.pressure << "\n";
  oss << "DENS=" << m_physics.density << "\n";
  oss << "VEL_U=" << m_physics.velocity_u << "\n";
  oss << "VEL_V=" << m_physics.velocity_v << "\n";

  // Material (just the name, loading will need to find the material)
  oss << "MAT=" << (m_material ? m_material->getName() : "") << "\n";

  // Vertex velocities
  oss << "VERTICES=";
  for (size_t i = 0; i < m_vertexVelocities.size(); ++i) {
    if (i > 0)
      oss << ",";
    oss << m_vertexVelocities[i];
  }
  oss << "\n";

  // Fixed properties
  oss << "PROPS=";
  for (size_t i = 0; i < static_cast<size_t>(PropertyType::COUNT); ++i) {
    if (i > 0)
      oss << ",";
    oss << m_fixedProperties[i];
  }
  oss << "\n";

  // Dynamic properties
  oss << "DYNPROPS=" << m_dynamicProperties.size() << "\n";
  for (const auto &[key, value] : m_dynamicProperties) {
    oss << key << "=" << value << "\n";
  }

  // Boundary condition information (if present)
  if (m_boundaryCondition) {
    oss << "BOUNDARY_TYPE=" << m_boundaryCondition->getType() << "\n";
    oss << "BOUNDARY_DATA=" << m_boundaryCondition->serialize() << "\n";
  }

  return oss.str();
}

// Complete implementation of the deserialize method with boundary support
bool Cell::deserialize(const std::string &data) {
  std::istringstream iss(data);
  std::string line, key, value;

  // Read version line
  std::getline(iss, line);
  if (line != "CELL_V1") {
    return false; // Unsupported version
  }

  // Parse key-value pairs
  while (std::getline(iss, line)) {
    size_t pos = line.find('=');
    if (pos == std::string::npos) {
      continue; // Skip lines without key-value format
    }

    key = line.substr(0, pos);
    value = line.substr(pos + 1);

    if (key == "TYPE") {
      setType(static_cast<CellType>(std::stoi(value)));
    } else if (key == "FIXED") {
      setFixed(std::stoi(value) != 0);
    } else if (key == "IS_BOUNDARY") {
      m_state.isBoundary = (std::stoi(value) != 0);
    } else if (key == "IS_OBSTACLE") {
      m_state.isObstacle = (std::stoi(value) != 0);
    } else if (key == "FLAGS") {
      m_state.flags = std::stoull(value);
    } else if (key == "TEMP") {
      setTemperature(std::stod(value));
    } else if (key == "PRES") {
      setPressure(std::stod(value));
    } else if (key == "DENS") {
      setDensity(std::stod(value));
    } else if (key == "VEL_U") {
      setVelocityU(std::stod(value));
    } else if (key == "VEL_V") {
      setVelocityV(std::stod(value));
    } else if (key == "MAT") {
      // Material handling - would need reference to material registry
      // This is typically handled at a higher level (e.g., by Grid)
      // We just store the name here for reference
      if (!value.empty()) {
        // In a real implementation, we would look up the material by name
        // m_material = m_grid->findMaterial(value);
      } else {
        m_material = nullptr;
      }
    } else if (key == "VERTICES") {
      std::istringstream vss(value);
      std::string component;
      size_t idx = 0;

      while (std::getline(vss, component, ',') &&
             idx < m_vertexVelocities.size()) {
        m_vertexVelocities[idx++] = std::stod(component);
      }
    } else if (key == "PROPS") {
      std::istringstream pss(value);
      std::string prop_val;
      size_t idx = 0;

      while (std::getline(pss, prop_val, ',') &&
             idx < static_cast<size_t>(PropertyType::COUNT)) {
        setProperty(static_cast<PropertyType>(idx), std::stod(prop_val));
        idx++;
      }
    } else if (key == "DYNPROPS") {
      // Next N lines contain dynamic properties
      int count = std::stoi(value);
      for (int i = 0; i < count && std::getline(iss, line); i++) {
        pos = line.find('=');
        if (pos != std::string::npos) {
          setDynamicProperty(line.substr(0, pos),
                             std::stod(line.substr(pos + 1)));
        }
      }
    } else if (key == "BOUNDARY_TYPE") {
      std::string boundaryType = value;

      // We need the next line which should contain the boundary data
      if (std::getline(iss, line)) {
        pos = line.find('=');
        if (pos != std::string::npos &&
            line.substr(0, pos) == "BOUNDARY_DATA") {
          std::string boundaryData = line.substr(pos + 1);

          // Use the BoundaryFactory to create and deserialize the boundary
          auto boundary =
              BoundaryFactory::deserializeBoundary(boundaryType, boundaryData);
          if (boundary) {
            setBoundaryCondition(boundary);
          }
        }
      }
    }
  }

  // Validate the deserialized data
  // Ensure state consistency
  if (m_state.type == CellType::BOUNDARY && !m_state.isBoundary) {
    m_state.isBoundary = true;
  }
  if (m_state.type == CellType::SOLID && !m_state.isObstacle) {
    m_state.isObstacle = true;
  }

  // Ensure fixed status is consistent with cell type
  if (m_state.type == CellType::BOUNDARY || m_state.type == CellType::SOLID) {
    m_state.isFixed = true;
  }

  // Ensure boundary and obstacle flags are set correctly
  if (m_state.isBoundary) {
    m_state.flags |= static_cast<uint64_t>(CellFlag::IS_BOUNDARY);
  }
  if (m_state.isObstacle) {
    m_state.flags |= static_cast<uint64_t>(CellFlag::IS_OBSTACLE);
  }

  return true;
}
