// Cell.cpp

#include "Cell.h"
#include "Mesh.h"
#include "Units.h"

#include <algorithm>
#include <cassert>
#include <iomanip>
#include <sstream>

// Initialize static property compute functions map
std::unordered_map<Cell::PropertyType, Cell::PropertyComputeFunction>
    Cell::s_propertyComputeFunctions;

//------------------------------------------------------------------------------
// Constructors
//------------------------------------------------------------------------------
Cell::Cell(Mesh *mesh, size_t i, size_t j) : m_mesh(mesh), m_i(i), m_j(j) {
  // No further initialization needed since data is stored in fields
}

Cell::Cell() : m_mesh(nullptr), m_i(0), m_j(0) {
  // Default constructor for temporary cells
}

Cell::Cell(Mesh *mesh, size_t i, size_t j, CellType type)
    : m_mesh(mesh), m_i(i), m_j(j) {
  // Set type directly
  setType(type);
}

//------------------------------------------------------------------------------
// Type and Flag Operations
//------------------------------------------------------------------------------
Cell::CellType Cell::getType() const {
  if (!m_mesh)
    return CellType::FLUID; // Default for detached cells

  // Access the cell type field in the mesh
  return m_mesh->getCellTypeField()(m_i, m_j);
}

void Cell::setType(CellType type) {
  if (!m_mesh)
    return; // No-op for detached cells

  // Update the cell type field
  m_mesh->getCellTypeField()(m_i, m_j) = type;

  // Update related flags based on type
  if (type == CellType::BOUNDARY) {
    setFixed(true);
    setBoundary(true);
    setObstacle(false);
  } else if (type == CellType::SOLID) {
    setFixed(true);
    setBoundary(false);
    setObstacle(true);
  } else { // FLUID
    setFixed(false);
    setBoundary(false);
    setObstacle(false);
  }
}

bool Cell::isFixed() const {
  if (!m_mesh)
    return false; // Default for detached cells

  // Access the fixed status field in the mesh
  return m_mesh->getFixedStatusField()(m_i, m_j);
}

void Cell::setFixed(bool fixed) {
  if (!m_mesh)
    return; // No-op for detached cells

  // Update the fixed status field
  m_mesh->getFixedStatusField()(m_i, m_j) = fixed;
}

bool Cell::isBoundary() const {
  if (!m_mesh)
    return false; // Default for detached cells

  // Access the boundary flag field in the mesh
  return m_mesh->getBoundaryFlagField()(m_i, m_j);
}

void Cell::setBoundary(bool isBoundary) {
  if (!m_mesh)
    return; // No-op for detached cells

  // Update the boundary flag field
  m_mesh->getBoundaryFlagField()(m_i, m_j) = isBoundary;

  // Update type if necessary
  if (isBoundary) {
    setType(CellType::BOUNDARY);
  } else if (getType() == CellType::BOUNDARY) {
    setType(CellType::FLUID);
  }
}

bool Cell::isObstacle() const {
  if (!m_mesh)
    return false; // Default for detached cells

  // Access the obstacle flag field in the mesh
  return m_mesh->getObstacleFlagField()(m_i, m_j);
}

void Cell::setObstacle(bool isObstacle) {
  if (!m_mesh)
    return; // No-op for detached cells

  // Update the obstacle flag field
  m_mesh->getObstacleFlagField()(m_i, m_j) = isObstacle;

  // Update type if necessary
  if (isObstacle) {
    setType(CellType::SOLID);
  } else if (getType() == CellType::SOLID) {
    setType(CellType::FLUID);
  }
}

bool Cell::getFlag(CellFlag flag) const {
  if (!m_mesh)
    return false; // Default for detached cells
  if (flag == CellFlag::COUNT)
    return false;

  // Access the flags field in the mesh
  uint64_t flags = m_mesh->getFlagsField()(m_i, m_j);
  return (flags & static_cast<uint64_t>(flag)) != 0;
}

void Cell::setFlag(CellFlag flag, bool value) {
  if (!m_mesh)
    return; // No-op for detached cells
  if (flag == CellFlag::COUNT)
    return;

  // Update the flags field
  uint64_t &flags = m_mesh->getFlagsField()(m_i, m_j);
  if (value)
    flags |= static_cast<uint64_t>(flag);
  else
    flags &= ~static_cast<uint64_t>(flag);
}

//------------------------------------------------------------------------------
// Cell-Centered Physical Property Accessors (SI Units)
//------------------------------------------------------------------------------
double Cell::getTemperature() const {
  if (!m_mesh)
    return 293.15; // Default for detached cells

  // Access the temperature field in the mesh
  return m_mesh->getTemperatureField()(m_i, m_j);
}

void Cell::setTemperature(double temperature) {
  if (!m_mesh)
    return; // No-op for detached cells

  // Update the temperature field with a valid value
  m_mesh->getTemperatureField()(m_i, m_j) =
      Units::enforceValidTemperature(temperature);
}

double Cell::getPressure() const {
  if (!m_mesh)
    return 101325.0; // Default for detached cells

  // Access the pressure field in the mesh
  return m_mesh->getPressureField()(m_i, m_j);
}

void Cell::setPressure(double pressure) {
  if (!m_mesh)
    return; // No-op for detached cells

  // Update the pressure field with a valid value
  m_mesh->getPressureField()(m_i, m_j) = Units::enforceValidPressure(pressure);
}

double Cell::getDensity() const {
  if (!m_mesh)
    return 1.0; // Default for detached cells

  // Access the density field in the mesh
  return m_mesh->getDensityField()(m_i, m_j);
}

void Cell::setDensity(double density) {
  if (!m_mesh)
    return; // No-op for detached cells

  // Update the density field with a valid value
  m_mesh->getDensityField()(m_i, m_j) = Units::enforceValidDensity(density);
}

double Cell::getVelocityU() const {
  if (!m_mesh)
    return 0.0; // Default for detached cells

  // Access the velocity-U field in the mesh
  return m_mesh->getVelocityUField()(m_i, m_j);
}

void Cell::setVelocityU(double vx) {
  if (!m_mesh)
    return; // No-op for detached cells

  // Update the velocity-U field
  m_mesh->getVelocityUField()(m_i, m_j) = vx;
}

double Cell::getVelocityV() const {
  if (!m_mesh)
    return 0.0; // Default for detached cells

  // Access the velocity-V field in the mesh
  return m_mesh->getVelocityVField()(m_i, m_j);
}

void Cell::setVelocityV(double vy) {
  if (!m_mesh)
    return; // No-op for detached cells

  // Update the velocity-V field
  m_mesh->getVelocityVField()(m_i, m_j) = vy;
}

//------------------------------------------------------------------------------
// Boundary condition access
//------------------------------------------------------------------------------
std::shared_ptr<BoundaryClass> Cell::getBoundaryCondition() const {
  if (!m_mesh)
    return nullptr; // Default for detached cells

  // Access the boundary condition in the mesh
  return m_mesh->getCellBoundaryCondition(m_i, m_j);
}

void Cell::setBoundaryCondition(std::shared_ptr<BoundaryClass> bc) {
  if (!m_mesh)
    return; // No-op for detached cells

  // Set the boundary condition in the mesh
  m_mesh->setCellBoundaryCondition(m_i, m_j, bc);
}

bool Cell::hasBoundaryCondition() const {
  return getBoundaryCondition() != nullptr;
}

//------------------------------------------------------------------------------
// Unit Conversion Methods
//------------------------------------------------------------------------------
void Cell::setTemperatureWithUnits(double temperature,
                                   const std::string &unit) {
  double kelvin = Units::convert(temperature, unit, "K");
  setTemperature(kelvin);
}

double Cell::getTemperatureWithUnits(const std::string &unit) const {
  return Units::convert(getTemperature(), "K", unit);
}

void Cell::setPressureWithUnits(double pressure, const std::string &unit) {
  double pascal = Units::convert(pressure, unit, "Pa");
  setPressure(pascal);
}

double Cell::getPressureWithUnits(const std::string &unit) const {
  return Units::convert(getPressure(), "Pa", unit);
}

void Cell::setDensityWithUnits(double density, const std::string &unit) {
  double siDensity = Units::convert(density, unit, "kg/m³");
  setDensity(siDensity);
}

double Cell::getDensityWithUnits(const std::string &unit) const {
  return Units::convert(getDensity(), "kg/m³", unit);
}

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

//------------------------------------------------------------------------------
// Vertex Velocity Accessors
//------------------------------------------------------------------------------
std::pair<double, double> Cell::getVertexVelocity(VertexPosition pos) const {
  if (!m_mesh)
    return {0.0, 0.0}; // Default for detached cells

  // Access the vertex velocities in the mesh
  size_t vertexIdx = static_cast<size_t>(pos);
  double vx = m_mesh->getVertexVelocityXField()(m_i, m_j, vertexIdx);
  double vy = m_mesh->getVertexVelocityYField()(m_i, m_j, vertexIdx);
  return {vx, vy};
}

void Cell::setVertexVelocity(VertexPosition pos, double vx, double vy) {
  if (!m_mesh)
    return; // No-op for detached cells

  // Update the vertex velocities in the mesh
  size_t vertexIdx = static_cast<size_t>(pos);
  m_mesh->getVertexVelocityXField()(m_i, m_j, vertexIdx) = vx;
  m_mesh->getVertexVelocityYField()(m_i, m_j, vertexIdx) = vy;
}

void Cell::setVertexVelocityWithUnits(VertexPosition pos, double vx, double vy,
                                      const std::string &unit) {
  double vx_mps = Units::convert(vx, unit, "m/s");
  double vy_mps = Units::convert(vy, unit, "m/s");
  setVertexVelocity(pos, vx_mps, vy_mps);
}

std::pair<double, double>
Cell::getVertexVelocityWithUnits(VertexPosition pos,
                                 const std::string &unit) const {
  auto [vx, vy] = getVertexVelocity(pos);
  return {Units::convert(vx, "m/s", unit), Units::convert(vy, "m/s", unit)};
}

//------------------------------------------------------------------------------
// Material Integration
//------------------------------------------------------------------------------
std::shared_ptr<Material> Cell::getMaterial() const {
  if (!m_mesh)
    return nullptr; // Default for detached cells

  // Access the material in the mesh
  return m_mesh->getCellMaterial(m_i, m_j);
}

void Cell::setMaterial(std::shared_ptr<Material> material) {
  if (!m_mesh)
    return; // No-op for detached cells

  // Set the material in the mesh
  m_mesh->setCellMaterial(m_i, m_j, material);
}

void Cell::mixMaterial(const std::shared_ptr<Material> &other,
                       double mixFraction, const std::string &mixingRule) {
  if (!m_mesh)
    return; // No-op for detached cells

  auto currentMaterial = getMaterial();
  if (currentMaterial) {
    setMaterial(currentMaterial->createMixture(other, mixFraction, mixingRule));
  } else {
    setMaterial(other);
  }
}

double Cell::getEffectiveDensity() const {
  double T = getTemperature();
  auto material = getMaterial();
  if (material) {
    return material->getPropertyAtTemperature(
        Material::MaterialProperty::DENSITY, T);
  }
  return getDensity();
}

//------------------------------------------------------------------------------
// Fixed Property Access (for PropertyType)
//------------------------------------------------------------------------------
void Cell::setProperty(PropertyType type, double value) {
  if (!m_mesh)
    return; // No-op for detached cells

  // Access the properties field in the mesh
  size_t propIdx = static_cast<size_t>(type);
  m_mesh->getPropertiesField()(m_i, m_j, propIdx) = value;
}

double Cell::getProperty(PropertyType type) const {
  if (!m_mesh)
    return 0.0; // Default for detached cells

  // Access the properties field in the mesh
  size_t propIdx = static_cast<size_t>(type);
  return m_mesh->getPropertiesField()(m_i, m_j, propIdx);
}

double Cell::getVorticity() const {
  return getProperty(PropertyType::VORTICITY);
}

void Cell::setVorticity(double value) {
  setProperty(PropertyType::VORTICITY, value);
}

double Cell::getStreamFunction() const {
  return getProperty(PropertyType::STREAM_FUNCTION);
}

void Cell::setStreamFunction(double value) {
  setProperty(PropertyType::STREAM_FUNCTION, value);
}

double Cell::getKineticEnergy() const {
  return getProperty(PropertyType::KINETIC_ENERGY);
}

void Cell::setKineticEnergy(double value) {
  setProperty(PropertyType::KINETIC_ENERGY, value);
}

double Cell::getDivergence() const {
  return getProperty(PropertyType::DIVERGENCE);
}

void Cell::setDivergence(double value) {
  setProperty(PropertyType::DIVERGENCE, value);
}

double Cell::getPressureCorrection() const {
  return getProperty(PropertyType::PRESSURE_CORRECTION);
}

void Cell::setPressureCorrection(double value) {
  setProperty(PropertyType::PRESSURE_CORRECTION, value);
}

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

double Cell::getShearStress() const {
  return getProperty(PropertyType::SHEAR_STRESS);
}

void Cell::setShearStress(double value) {
  setProperty(PropertyType::SHEAR_STRESS, value);
}

double Cell::getWallDistance() const {
  return getProperty(PropertyType::WALL_DISTANCE);
}

void Cell::setWallDistance(double value) {
  setProperty(PropertyType::WALL_DISTANCE, value);
}

//------------------------------------------------------------------------------
// Dynamic Properties
//------------------------------------------------------------------------------
void Cell::setDynamicProperty(const std::string &name, double value) {
  // Dynamic properties are stored locally in the Cell, not in Fields
  m_dynamicProperties[name] = value;
}

double Cell::getDynamicProperty(const std::string &name,
                                double defaultValue) const {
  auto it = m_dynamicProperties.find(name);
  return (it != m_dynamicProperties.end()) ? it->second : defaultValue;
}

//------------------------------------------------------------------------------
// Property Computation Registration
//------------------------------------------------------------------------------
void Cell::registerPropertyComputation(PropertyType type,
                                       PropertyComputeFunction func) {
  s_propertyComputeFunctions[type] = func;
}

Cell::PropertyComputeFunction Cell::getPropertyComputation(PropertyType type) {
  auto it = s_propertyComputeFunctions.find(type);
  if (it != s_propertyComputeFunctions.end()) {
    return it->second;
  }
  // Return a no-op function if no computation is registered
  return [](Cell &, const std::array<Cell *, 4> *) {};
}

void Cell::computeDerivedProperties(const std::array<Cell *, 4> *neighbors) {
  // Execute all registered property computations
  for (size_t i = 0; i < static_cast<size_t>(PropertyType::COUNT); ++i) {
    PropertyType type = static_cast<PropertyType>(i);
    auto it = s_propertyComputeFunctions.find(type);
    if (it != s_propertyComputeFunctions.end()) {
      it->second(*this, neighbors);
    }
  }
}

//------------------------------------------------------------------------------
// Reset: resets cell state to default values.
//------------------------------------------------------------------------------
void Cell::reset() {
  if (!m_mesh)
    return; // No-op for detached cells

  // Preserve type, boundary and obstacle flags
  CellType type = getType();
  bool isBoundary = this->isBoundary();
  bool isObstacle = this->isObstacle();

  // Reset physical properties to defaults
  setTemperature(293.15);
  setPressure(101325.0);
  setDensity(1.0);
  setVelocityU(0.0);
  setVelocityV(0.0);

  // Reset vertex velocities
  for (size_t idx = 0; idx < 4; ++idx) {
    VertexPosition pos = static_cast<VertexPosition>(idx);
    setVertexVelocity(pos, 0.0, 0.0);
  }

  // Reset fixed properties
  for (size_t i = 0; i < static_cast<size_t>(PropertyType::COUNT); ++i) {
    setProperty(static_cast<PropertyType>(i), 0.0);
  }

  // Clear dynamic properties
  m_dynamicProperties.clear();

  // Restore type and flags
  setType(type);
  setBoundary(isBoundary);
  setObstacle(isObstacle);

  // Note: Material is preserved
}

//------------------------------------------------------------------------------
// Serialization
//------------------------------------------------------------------------------
std::string Cell::serialize() const {
  std::ostringstream oss;
  oss << "CELL_V1\n";
  oss << "TYPE=" << static_cast<int>(getType()) << "\n";
  oss << "FIXED=" << (isFixed() ? 1 : 0) << "\n";
  oss << "IS_BOUNDARY=" << (isBoundary() ? 1 : 0) << "\n";
  oss << "IS_OBSTACLE=" << (isObstacle() ? 1 : 0) << "\n";
  oss << "FLAGS=" << m_mesh->getFlagsField()(m_i, m_j) << "\n";
  oss << "TEMP=" << getTemperature() << "\n";
  oss << "PRES=" << getPressure() << "\n";
  oss << "DENS=" << getDensity() << "\n";
  oss << "VEL_U=" << getVelocityU() << "\n";
  oss << "VEL_V=" << getVelocityV() << "\n";

  // For simplicity, we store material name.
  auto material = getMaterial();
  oss << "MAT=" << (material ? material->getName() : "") << "\n";

  // Serialize vertex velocities.
  oss << "VERTICES=";
  for (size_t i = 0; i < 4; ++i) {
    if (i > 0)
      oss << ",";
    auto [vx, vy] = getVertexVelocity(static_cast<VertexPosition>(i));
    oss << vx << ";" << vy;
  }
  oss << "\n";

  // Serialize fixed properties.
  oss << "PROPS=";
  for (size_t i = 0; i < static_cast<size_t>(PropertyType::COUNT); ++i) {
    if (i > 0)
      oss << ",";
    oss << getProperty(static_cast<PropertyType>(i));
  }
  oss << "\n";

  // Serialize dynamic properties.
  oss << "DYNPROPS=" << m_dynamicProperties.size() << "\n";
  for (const auto &kv : m_dynamicProperties) {
    oss << kv.first << "=" << kv.second << "\n";
  }
  return oss.str();
}

bool Cell::deserialize(const std::string &data) {
  if (!m_mesh)
    return false; // Cannot deserialize without a mesh

  std::istringstream iss(data);
  std::string line;
  std::getline(iss, line);
  if (line != "CELL_V1")
    return false;

  while (std::getline(iss, line)) {
    size_t pos = line.find('=');
    if (pos == std::string::npos)
      continue;

    std::string key = line.substr(0, pos);
    std::string value = line.substr(pos + 1);

    if (key == "TYPE") {
      setType(static_cast<CellType>(std::stoi(value)));
    } else if (key == "FIXED") {
      setFixed(std::stoi(value) != 0);
    } else if (key == "IS_BOUNDARY") {
      setBoundary(std::stoi(value) != 0);
    } else if (key == "IS_OBSTACLE") {
      setObstacle(std::stoi(value) != 0);
    } else if (key == "FLAGS") {
      m_mesh->getFlagsField()(m_i, m_j) = std::stoull(value);
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
      // Material assignment to be handled externally via a registry.
      // This would typically be handled by the Mesh after deserialization.
    } else if (key == "VERTICES") {
      std::istringstream viss(value);
      std::string component;
      size_t idx = 0;
      while (std::getline(viss, component, ',') && idx < 4) {
        size_t semicolonPos = component.find(';');
        if (semicolonPos != std::string::npos) {
          double vx = std::stod(component.substr(0, semicolonPos));
          double vy = std::stod(component.substr(semicolonPos + 1));
          setVertexVelocity(static_cast<VertexPosition>(idx), vx, vy);
        }
        idx++;
      }
    } else if (key == "PROPS") {
      std::istringstream pss(value);
      std::string propVal;
      size_t idx = 0;
      while (std::getline(pss, propVal, ',') &&
             idx < static_cast<size_t>(PropertyType::COUNT)) {
        setProperty(static_cast<PropertyType>(idx), std::stod(propVal));
        idx++;
      }
    } else if (key == "DYNPROPS") {
      int count = std::stoi(value);
      m_dynamicProperties.clear();
      for (int i = 0; i < count && std::getline(iss, line); ++i) {
        size_t eqPos = line.find('=');
        if (eqPos != std::string::npos) {
          std::string propName = line.substr(0, eqPos);
          double propValue = std::stod(line.substr(eqPos + 1));
          m_dynamicProperties[propName] = propValue;
        }
      }
    }
  }
  return true;
}

//------------------------------------------------------------------------------
// Diagnostic and Visualization Functions
//------------------------------------------------------------------------------
void Cell::print(std::ostream &os, int verbosity) const {
  os << toString(verbosity);
}

std::string Cell::toString(int verbosity) const {
  std::ostringstream oss;
  oss << "Cell [Type: " << cellTypeToString(getType()) << "]";
  if (m_mesh) {
    oss << " at grid position (" << m_i << ", " << m_j << ")";
  } else {
    oss << " [DETACHED]";
  }
  oss << "\n";

  oss << "  Fixed: " << (isFixed() ? "Yes" : "No") << "\n";
  oss << "  Temperature: " << getTemperature() << " K ("
      << getTemperatureWithUnits("C") << " °C)\n";
  oss << "  Pressure: " << getPressure() << " Pa ("
      << getPressureWithUnits("bar") << " bar)\n";
  oss << "  Density: " << getDensity() << " kg/m³\n";
  oss << "  Velocity: [" << getVelocityU() << ", " << getVelocityV()
      << "] m/s\n";

  auto material = getMaterial();
  if (material)
    oss << "  Material: " << material->getName() << "\n";
  else
    oss << "  Material: None\n";

  if (verbosity >= 2) {
    oss << "  Properties:\n";
    oss << "    Vorticity: " << getVorticity() << "\n";
    oss << "    Stream Function: " << getStreamFunction() << "\n";
    oss << "    Kinetic Energy: " << getKineticEnergy() << "\n";
    oss << "    Divergence: " << getDivergence() << "\n";
    oss << "    Pressure Correction: " << getPressureCorrection() << "\n";
    oss << "    Heat Flux X: " << getHeatFluxX() << "\n";
    oss << "    Heat Flux Y: " << getHeatFluxY() << "\n";
    oss << "    Shear Stress: " << getShearStress() << "\n";
    oss << "    Wall Distance: " << getWallDistance() << "\n";

    if (!m_dynamicProperties.empty()) {
      oss << "  Dynamic Properties:\n";
      for (const auto &kv : m_dynamicProperties)
        oss << "    " << kv.first << ": " << kv.second << "\n";
    }
  }
  return oss.str();
}

std::string Cell::toSVG(double scale, bool showVelocity) const {
  std::ostringstream svg;
  // For simplicity, assume a fixed cell size (e.g., 10 m) scaled by 'scale'
  double cellSize = 10.0;
  double x =
      0.0; // Physical coordinates would be computed based on grid indices
  double y = 0.0;
  if (m_mesh) {
    x = m_mesh->physicalX(m_i) * scale;
    y = m_mesh->physicalY(m_j) * scale;
    cellSize = std::min(m_mesh->getDx(), m_mesh->getDy()) * scale;
  }

  svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << cellSize + 10
      << "\" height=\"" << cellSize + 10 << "\">\n";

  // Color based on cell type
  std::string fillColor = "rgb(200,230,255)"; // Default fluid color
  if (getType() == CellType::SOLID) {
    fillColor = "rgb(150,150,150)"; // Gray for solid
  } else if (getType() == CellType::BOUNDARY) {
    fillColor = "rgb(255,200,200)"; // Light red for boundary
  }

  svg << "  <rect x=\"" << x << "\" y=\"" << y << "\" width=\"" << cellSize
      << "\" height=\"" << cellSize << "\" fill=\"" << fillColor
      << "\" stroke=\"black\" stroke-width=\"1\" />\n";

  if (showVelocity) {
    double vx = getVelocityU();
    double vy = getVelocityV();
    double mag = std::sqrt(vx * vx + vy * vy);
    if (mag > 1e-6) {
      double cx = x + cellSize / 2;
      double cy = y + cellSize / 2;
      vx = vx / mag;
      vy = vy / mag;
      double arrowLen = cellSize * 0.4;
      double ex = cx + vx * arrowLen;
      double ey = cy + vy * arrowLen;
      svg << "  <line x1=\"" << cx << "\" y1=\"" << cy << "\" x2=\"" << ex
          << "\" y2=\"" << ey << "\" stroke=\"black\" stroke-width=\"2\" />\n";

      // Add arrow head
      double headSize = cellSize * 0.1;
      double angle = std::atan2(vy, vx);
      double x1 = ex - headSize * std::cos(angle + M_PI / 6);
      double y1 = ey - headSize * std::sin(angle + M_PI / 6);
      double x2 = ex - headSize * std::cos(angle - M_PI / 6);
      double y2 = ey - headSize * std::sin(angle - M_PI / 6);

      svg << "  <line x1=\"" << ex << "\" y1=\"" << ey << "\" x2=\"" << x1
          << "\" y2=\"" << y1 << "\" stroke=\"black\" stroke-width=\"2\" />\n";
      svg << "  <line x1=\"" << ex << "\" y1=\"" << ey << "\" x2=\"" << x2
          << "\" y2=\"" << y2 << "\" stroke=\"black\" stroke-width=\"2\" />\n";
    }
  }

  svg << "</svg>\n";
  return svg.str();
}

//------------------------------------------------------------------------------
// Static Helper Methods
//------------------------------------------------------------------------------
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
  default:
    return "UNKNOWN";
  }
}
