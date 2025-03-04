#include "Cell.h"
#include <algorithm>
#include <cmath>

// Forward declaration if needed
#include "Material.h"

// Static hash function for property names
uint32_t Cell::hashName(const std::string &name) {
  // Simple hash function (could use std::hash)
  uint32_t hash = 0;
  for (char c : name) {
    hash = hash * 31 + c;
  }
  return hash;
}

Cell::Cell(Grid *grid)
    : m_grid(grid), m_boundaryCondition(nullptr), m_type(CellType::FLUID),
      m_temperature(293.15), m_pressure(101325.0), m_density(1.0),
      m_material(nullptr), m_isFixed(false), m_is_obstacle(false),
      m_is_boundary(false), m_velocity_x(0.0), m_velocity_y(0.0) {
  // Initialize vertices with zero velocity
  for (auto &vertex : m_vertices) {
    vertex.vx = 0.0;
    vertex.vy = 0.0;
  }

  // Initialize fixed properties to zero
  m_fixedProperties.fill(0.0);

  // Initialize dynamic properties map with some capacity
  m_dynamicProperties.reserve(8);
}

Cell::Cell(CellType type, Grid *grid)
    : m_grid(grid), m_boundaryCondition(nullptr), m_type(type),
      m_temperature(293.15), m_pressure(101325.0), m_density(1.0),
      m_material(nullptr),
      m_isFixed(type == CellType::SOLID || type == CellType::BOUNDARY),
      m_is_obstacle(type == CellType::SOLID),
      m_is_boundary(type == CellType::BOUNDARY), m_velocity_x(0.0),
      m_velocity_y(0.0) {
  // Initialize vertices with zero velocity
  for (auto &vertex : m_vertices) {
    vertex.vx = 0.0;
    vertex.vy = 0.0;
  }

  // Initialize fixed properties to zero
  m_fixedProperties.fill(0.0);

  // Initialize dynamic properties map with some capacity
  m_dynamicProperties.reserve(8);

  // Set appropriate flags based on cell type
  if (type == CellType::BOUNDARY) {
    setFlag(CellFlag::IS_WALL, true); // Default boundary is a wall
    setFlag(CellFlag::IS_BOUNDARY, true);
  } else if (type == CellType::SOLID) {
    setFlag(CellFlag::IS_OBSTACLE, true);
  }
}

Cell::CellType Cell::getType() const { return m_type; }

void Cell::setType(CellType type) {
  m_type = type;
  // When setting a cell to SOLID or BOUNDARY, it's typically fixed
  if (type == CellType::SOLID || type == CellType::BOUNDARY) {
    m_isFixed = true;

    if (type == CellType::BOUNDARY) {
      setFlag(CellFlag::IS_BOUNDARY, true);
    } else if (type == CellType::SOLID) {
      setFlag(CellFlag::IS_OBSTACLE, true);
    }
  }
}

double Cell::getTemperature() const { return m_temperature; }

void Cell::setTemperature(double temperature) {
  m_temperature = Units::enforceValidTemperature(temperature);
}

double Cell::getPressure() const { return m_pressure; }

void Cell::setPressure(double pressure) {
  m_pressure = Units::enforceValidPressure(pressure);
}

  double Cell::getDensity() const { return m_density; }

  void Cell::setDensity(double density) {
    m_density = Units::enforceValidDensity(density);
  }

  std::shared_ptr<Material> Cell::getMaterial() const { return m_material; }

  void Cell::setMaterial(std::shared_ptr<Material> material) {
    m_material = material;
  }

  // Additional method for Grid compatibility using raw pointers
  void Cell::setMaterial(Material *material) {
    if (material != nullptr) {
      // Create a shared_ptr that doesn't delete the Material (Grid owns it)
      m_material = std::shared_ptr<Material>(material, [](Material *) {});
    } else {
      m_material = nullptr;
    }
  }

  bool Cell::isFixed() const { return m_isFixed; }

  void Cell::setFixed(bool fixed) { m_isFixed = fixed; }

  std::pair<double, double>
  Cell::getVertexVelocity(VertexPosition position) const {
    const auto &vertex = m_vertices[static_cast<size_t>(position)];
    return {vertex.vx, vertex.vy};
  }

  void Cell::setVertexVelocity(VertexPosition position, double vx, double vy) {
    auto &vertex = m_vertices[static_cast<size_t>(position)];
    vertex.vx = vx;
    vertex.vy = vy;
  }

  Cell::Vertex &Cell::getVertex(VertexPosition position) {
    return m_vertices[static_cast<size_t>(position)];
  }

  const Cell::Vertex &Cell::getVertex(VertexPosition position) const {
    return m_vertices[static_cast<size_t>(position)];
  }

  void Cell::setProperty(PropertyType type, double value) {
    m_fixedProperties[static_cast<size_t>(type)] = value;
  }

  double Cell::getProperty(PropertyType type) const {
    return m_fixedProperties[static_cast<size_t>(type)];
  }

  void Cell::setFlag(CellFlag flag, bool value) {
    if (value) {
      m_flags |= (1U << static_cast<uint32_t>(flag));
    } else {
      m_flags &= ~(1U << static_cast<uint32_t>(flag));
    }
  }

  bool Cell::getFlag(CellFlag flag) const {
    return (m_flags & (1U << static_cast<uint32_t>(flag))) != 0;
  }

  void Cell::setDynamicProperty(const std::string &name, double value) {
    m_dynamicProperties[name] = value;
  }

  double Cell::getDynamicProperty(const std::string &name,
                                  double defaultValue) const {
    auto it = m_dynamicProperties.find(name);
    return (it != m_dynamicProperties.end()) ? it->second : defaultValue;
  }

  void Cell::reset() {
    // Reset to default values
    m_temperature = 293.15;
    m_pressure = 101325.0;
    m_density = 1.0;
    m_velocity_x = 0.0;
    m_velocity_y = 0.0;

    // Reset velocities for vertices
    for (auto &vertex : m_vertices) {
      vertex.vx = 0.0;
      vertex.vy = 0.0;
    }

    // Reset fixed properties
    m_fixedProperties.fill(0.0);

    // Reset flags
    m_flags = 0;

    // Clear dynamic properties
    m_dynamicProperties.clear();

    // Keep type and material as they are
  }

  // Additional methods for Grid compatibility

  void Cell::setBoundary(bool isBoundary) {
    m_is_boundary = isBoundary;
    setFlag(CellFlag::IS_BOUNDARY, isBoundary);
    if (isBoundary) {
      setType(CellType::BOUNDARY);
    }
  }

  bool Cell::isBoundary() const { return m_is_boundary; }

  void Cell::setObstacle(bool isObstacle) {
    m_is_obstacle = isObstacle;
    setFlag(CellFlag::IS_OBSTACLE, isObstacle);
    if (isObstacle) {
      setType(CellType::SOLID);
    }
  }

  bool Cell::isObstacle() const { return m_is_obstacle; }

  double Cell::getVelocityX() const { return m_velocity_x; }

  double Cell::getVelocityY() const { return m_velocity_y; }

  void Cell::setVelocityX(double vx) { m_velocity_x = vx; }

  void Cell::setVelocityY(double vy) { m_velocity_y = vy; }
