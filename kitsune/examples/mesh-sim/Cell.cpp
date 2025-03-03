#include "Cell.h"
#include <cmath>
#include <algorithm>

// Static hash function for property names
uint32_t Cell::hashName(const std::string& name) {
  // Simple hash function (could use std::hash)
  uint32_t hash = 0;
  for (char c : name) {
    hash = hash * 31 + c;
  }
  return hash;
}

Cell::Cell() 
  : m_type(CellType::FLUID),
    m_temperature(293.15),  // Default to room temperature in Kelvin
    m_pressure(101325.0),   // Default to standard atmospheric pressure in Pascal
    m_density(1.0),         // Default to unit density
    m_isFixed(false) {
    
  // Initialize vertices with zero velocity
  for (auto& vertex : m_vertices) {
    vertex.vx = 0.0;
    vertex.vy = 0.0;
  }
    
  // Initialize fixed properties to zero
  m_fixedProperties.fill(0.0);
    
  // Initialize dynamic properties map with some capacity
  m_dynamicProperties.reserve(8);
}

Cell::Cell(CellType type) 
  : m_type(type),
    m_temperature(293.15),
    m_pressure(101325.0),
    m_density(1.0),
    m_isFixed(type == CellType::SOLID || type == CellType::BOUNDARY) {
    
  // Initialize vertices with zero velocity
  for (auto& vertex : m_vertices) {
    vertex.vx = 0.0;
    vertex.vy = 0.0;
  }
    
  // Initialize fixed properties to zero
  m_fixedProperties.fill(0.0);
    
  // Initialize dynamic properties map with some capacity
  m_dynamicProperties.reserve(8);
    
  // Set appropriate flags based on cell type
  if (type == CellType::BOUNDARY) {
    setFlag(CellFlag::IS_WALL, true);  // Default boundary is a wall
  }
}

Cell::CellType Cell::getType() const {
  return m_type;
}

void Cell::setType(CellType type) {
  m_type = type;
  // When setting a cell to SOLID or BOUNDARY, it's typically fixed
  if (type == CellType::SOLID || type == CellType::BOUNDARY) {
    m_isFixed = true;
  }
}

double Cell::getTemperature() const {
  return m_temperature;
}

void Cell::setTemperature(double temperature) {
  m_temperature = temperature;
}

double Cell::getPressure() const {
  return m_pressure;
}

void Cell::setPressure(double pressure) {
  m_pressure = pressure;
}

double Cell::getDensity() const {
  return m_density;
}

void Cell::setDensity(double density) {
  m_density = density;
}

std::shared_ptr<Material> Cell::getMaterial() const {
  return m_material;
}

void Cell::setMaterial(std::shared_ptr<Material> material) {
  m_material = material;
}

bool Cell::isFixed() const {
  return m_isFixed;
}

void Cell::setFixed(bool fixed) {
    m_isFixed = fixed;
}

std::pair<double, double> Cell::getVertexVelocity(VertexPosition position) const {
  const auto& vertex = m_vertices[static_cast<size_t>(position)];
  return {vertex.vx, vertex.vy};
}

void Cell::setVertexVelocity(VertexPosition position, double vx, double vy) {
  auto& vertex = m_vertices[static_cast<size_t>(position)];
  vertex.vx = vx;
  vertex.vy = vy;
}

Cell::Vertex& Cell::getVertex(VertexPosition position) {
  return m_vertices[static_cast<size_t>(position)];
}

const Cell::Vertex& Cell::getVertex(VertexPosition position) const {
  return m_vertices[static_cast<size_t>(position)];
}

void Cell::setProperty(PropertyType type, double value) {
  m_fixedProperties[static_cast<size_t>(type)] = value;
}

double Cell::getProperty(PropertyType type, double defaultValue) const {
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

void Cell::setDynamicProperty(const std::string& name, double value) {
  m_dynamicProperties[name] = value;
}

double Cell::getDynamicProperty(const std::string& name, double defaultValue) const {
  auto it = m_dynamicProperties.find(name);
  return (it != m_dynamicProperties.end()) ? it->second : defaultValue;
}

void Cell::reset() {
  // Reset to default values
  m_temperature = 293.15;
  m_pressure = 101325.0;
  m_density = 1.0;
    
  // Reset velocities
  for (auto& vertex : m_vertices) {
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


