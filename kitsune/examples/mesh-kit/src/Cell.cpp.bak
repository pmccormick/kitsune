#include "Cell.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>

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

  // Handle fixed state based on type
  setFixed(type == CellType::BOUNDARY || type == CellType::SOLID);

  // Update boundary/obstacle flags based on type
  if (type == CellType::BOUNDARY) {
    m_flags |= static_cast<uint32_t>(CellFlag::IS_BOUNDARY);
    m_flags &= ~static_cast<uint32_t>(CellFlag::IS_OBSTACLE);
  } else if (type == CellType::SOLID) {
    m_flags |= static_cast<uint32_t>(CellFlag::IS_OBSTACLE);
    m_flags &= ~static_cast<uint32_t>(CellFlag::IS_BOUNDARY);
  } else { // FLUID
    m_flags &= ~static_cast<uint32_t>(CellFlag::IS_BOUNDARY);
    m_flags &= ~static_cast<uint32_t>(CellFlag::IS_OBSTACLE);
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
    // Skip COUNT which isn't a real flag
    if (flag == CellFlag::COUNT)
      return;

    if (value) {
      // Handle mutually exclusive flags
      if (flag == CellFlag::IS_INLET) {
        // Clear outlet flag if setting inlet
        m_flags &= ~static_cast<uint32_t>(CellFlag::IS_OUTLET);
      } else if (flag == CellFlag::IS_OUTLET) {
        // Clear inlet flag if setting outlet
        m_flags &= ~static_cast<uint32_t>(CellFlag::IS_INLET);
      }

      // Set the requested flag
      m_flags |= static_cast<uint32_t>(flag);
    } else {
      // Clear the requested flag
      m_flags &= ~static_cast<uint32_t>(flag);
    }
  }

  bool Cell::getFlag(CellFlag flag) const {
    // Skip COUNT which isn't a real flag
    if (flag == CellFlag::COUNT)
      return false;

    return (m_flags & static_cast<uint32_t>(flag)) != 0;
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

  void Cell::setObstacle(bool obstacleFlag) {
    m_is_obstacle = obstacleFlag;
    setFlag(CellFlag::IS_OBSTACLE, obstacleFlag);
    if (obstacleFlag) {
      setType(CellType::SOLID);
    }
  }

  bool Cell::isObstacle() const { return m_is_obstacle; }

  double Cell::getVelocityX() const { return m_velocity_x; }

  double Cell::getVelocityY() const { return m_velocity_y; }

  void Cell::setVelocityX(double vx) { m_velocity_x = vx; }

  void Cell::setVelocityY(double vy) { m_velocity_y = vy; }

  // Static initialization of the property computation map
  std::unordered_map<Cell::PropertyType, Cell::PropertyComputeFunction>
      Cell::s_propertyComputations;

  void Cell::registerPropertyComputation(PropertyType type,
                                         PropertyComputeFunction computeFunc) {
    s_propertyComputations[type] = computeFunc;
  }

  void Cell::computeDerivedProperties(const std::array<Cell *, 4> *neighbors) {
    // Apply registered computation functions
    for (const auto &[type, computeFunc] : s_propertyComputations) {
      computeFunc(*this, neighbors);
    }
  }

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

  void Cell::print(std::ostream &os, int verbosity) const {
    os << "Cell [Type: " << cellTypeToString(m_type) << "]" << std::endl;

    if (verbosity >= 1) {
      // Basic properties
      os << "  Temperature: " << m_temperature << " K ("
         << getTemperatureWithUnits("C") << " °C)" << std::endl;
      os << "  Pressure: " << m_pressure << " Pa ("
         << getPressureWithUnits("bar") << " bar)" << std::endl;
      os << "  Density: " << m_density << " kg/m³" << std::endl;
      os << "  Velocity: [" << m_velocity_x << ", " << m_velocity_y << "] m/s"
         << std::endl;

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
      os << "    Kinetic Energy: " << getKineticEnergy() << " J/kg"
         << std::endl;
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

  std::string Cell::toString(int verbosity) const {
    std::ostringstream oss;
    print(oss, verbosity);
    return oss.str();
  }

  std::string Cell::serialize() const {
    std::ostringstream oss;

    // Version identifier to support future changes
    oss << "CELL_V1\n";

    // Core properties
    oss << "TYPE=" << static_cast<int>(m_type) << "\n";
    oss << "TEMP=" << m_temperature << "\n";
    oss << "PRES=" << m_pressure << "\n";
    oss << "DENS=" << m_density << "\n";
    oss << "VEL_X=" << m_velocity_x << "\n";
    oss << "VEL_Y=" << m_velocity_y << "\n";
    oss << "FIXED=" << (m_isFixed ? 1 : 0) << "\n";
    oss << "FLAGS=" << m_flags << "\n";

    // Material (just the name, loading will need to find the material)
    oss << "MAT=" << (m_material ? m_material->getName() : "") << "\n";

    // Vertex velocities
    for (int i = 0; i < 4; i++) {
      oss << "VERT" << i << "=" << m_vertices[i].vx << "," << m_vertices[i].vy
          << "\n";
    }

    // Properties
    oss << "PROPS=";
    for (size_t i = 0; i < static_cast<size_t>(PropertyType::COUNT); i++) {
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

    return oss.str();
  }

  bool Cell::deserialize(const std::string &data) {
    std::istringstream iss(data);
    std::string line, key, value;

    // Read version line
    std::getline(iss, line);
    if (line != "CELL_V1")
      return false;

    // Parse key-value pairs
    while (std::getline(iss, line)) {
      size_t pos = line.find('=');
      if (pos == std::string::npos)
        continue;

      key = line.substr(0, pos);
      value = line.substr(pos + 1);

      if (key == "TYPE") {
        setType(static_cast<CellType>(std::stoi(value)));
      } else if (key == "TEMP") {
        setTemperature(std::stod(value));
      } else if (key == "PRES") {
        setPressure(std::stod(value));
      } else if (key == "DENS") {
        setDensity(std::stod(value));
      } else if (key == "VEL_X") {
        setVelocityX(std::stod(value));
      } else if (key == "VEL_Y") {
        setVelocityY(std::stod(value));
      } else if (key == "FIXED") {
        setFixed(std::stoi(value) != 0);
      } else if (key == "FLAGS") {
        m_flags = std::stoull(value);
      } else if (key == "MAT") {
        // Material handling - would need reference to material registry
        // This would be handled at a higher level
      } else if (key.substr(0, 4) == "VERT") {
        int idx = std::stoi(key.substr(4, 1));
        pos = value.find(',');
        if (pos != std::string::npos) {
          double vx = std::stod(value.substr(0, pos));
          double vy = std::stod(value.substr(pos + 1));
          setVertexVelocity(static_cast<VertexPosition>(idx), vx, vy);
        }
      } else if (key == "PROPS") {
        std::istringstream props_ss(value);
        std::string prop_val;
        size_t idx = 0;

        while (std::getline(props_ss, prop_val, ',') &&
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
      }
    }

    return true;
  }

  /**
   * @brief Generate SVG representation of the cell
   * @param scale Scale factor for the rendering
   * @param showVelocity Whether to show velocity vectors
   * @return SVG string representation
   */
  std::string Cell::toSVG(double scale, bool showVelocity) const {
    std::ostringstream svg;

    // SVG header
    svg << "<svg width=\"" << 12 * scale << "\" height=\"" << 12 * scale
        << "\" xmlns=\"http://www.w3.org/2000/svg\">\n";

    // Cell background based on type
    std::string fillColor;
    switch (m_type) {
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
        << 10 * scale << "\" height=\"" << 10 * scale << "\" fill=\""
        << fillColor << "\" stroke=\"black\" stroke-width=\"1\"/>\n";

    // Draw temperature indication (red-blue gradient)
    double normTemp =
        (m_temperature - 273.15) / 100.0; // Normalize around 0°C = 273.15K
    normTemp = std::max(0.0, std::min(1.0, normTemp)); // Clamp to [0,1]
    int red = static_cast<int>(255 * normTemp);
    int blue = static_cast<int>(255 * (1.0 - normTemp));
    svg << "  <circle cx=\"" << 3 * scale << "\" cy=\"" << 3 * scale
        << "\" r=\"" << scale << "\" fill=\"rgb(" << red << ",0," << blue
        << ")\"/>\n";

    // Draw pressure indication (size of circle)
    double normPressure = m_pressure / 101325.0; // Normalize around 1 atm
    normPressure =
        std::max(0.2, std::min(1.5, normPressure)); // Clamp and scale
    svg << "  <circle cx=\"" << 8 * scale << "\" cy=\"" << 3 * scale
        << "\" r=\"" << scale * normPressure
        << "\" fill=\"rgba(0,0,0,0.3)\"/>\n";

    if (showVelocity) {
      // Draw velocity vector at center
      double velMag =
          std::sqrt(m_velocity_x * m_velocity_x + m_velocity_y * m_velocity_y);
      if (velMag > 1e-6) {             // Only if non-zero
        double velScale = 3.0 * scale; // Scale factor for velocity arrows
        double normVelX = m_velocity_x / velMag;
        double normVelY = m_velocity_y / velMag;

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
    /* For future consideration...
    // Register function to compute vorticity for structured grids
    void registerStructuredGridFunctions() {
      Cell::registerPropertyComputation(Cell::PropertyType::VORTICITY,
          [](Cell& cell, const std::array<Cell*, 4>* neighbors) {
              // Structured grid vorticity calculation
              // ...
          }
      );
    }

    // Register function to compute vorticity for unstructured grids
    void registerUnstructuredGridFunctions() {
      Cell::registerPropertyComputation(Cell::PropertyType::VORTICITY,
          [](Cell& cell, const std::array<Cell*, 4>* neighbors) {
              // Unstructured grid vorticity calculation
              // ...
          }
      );
    }
    */
