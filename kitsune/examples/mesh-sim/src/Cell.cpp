// Cell.cpp

#include "Cell.h"
#include "Units.h"

#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cassert>

//------------------------------------------------------------------------------
// Constructors
//------------------------------------------------------------------------------
Cell::Cell() {
    // Defaults already initialized.
}

Cell::Cell(CellType type) : m_type(type) {
    // Set fixed state and flags based on type.
    if (type == CellType::BOUNDARY || type == CellType::SOLID) {
        m_fixed = true;
        if (type == CellType::BOUNDARY) {
            m_isBoundary = true;
            m_flags |= static_cast<uint64_t>(CellFlag::IS_BOUNDARY);
        } else if (type == CellType::SOLID) {
            m_isObstacle = true;
            m_flags |= static_cast<uint64_t>(CellFlag::IS_OBSTACLE);
        }
    }
}

//------------------------------------------------------------------------------
// Type and Flag Operations
//------------------------------------------------------------------------------
Cell::CellType Cell::getType() const {
    return m_type;
}

void Cell::setType(CellType type) {
    m_type = type;
    // Update fixed and flag states.
    if (type == CellType::BOUNDARY) {
        m_fixed = true;
        m_isBoundary = true;
        m_isObstacle = false;
        m_flags |= static_cast<uint64_t>(CellFlag::IS_BOUNDARY);
        m_flags &= ~static_cast<uint64_t>(CellFlag::IS_OBSTACLE);
    } else if (type == CellType::SOLID) {
        m_fixed = true;
        m_isBoundary = false;
        m_isObstacle = true;
        m_flags |= static_cast<uint64_t>(CellFlag::IS_OBSTACLE);
        m_flags &= ~static_cast<uint64_t>(CellFlag::IS_BOUNDARY);
    } else { // FLUID
        m_fixed = false;
        m_isBoundary = false;
        m_isObstacle = false;
        m_flags &= ~static_cast<uint64_t>(CellFlag::IS_BOUNDARY);
        m_flags &= ~static_cast<uint64_t>(CellFlag::IS_OBSTACLE);
    }
}

bool Cell::isFixed() const {
    return m_fixed;
}

void Cell::setFixed(bool fixed) {
    m_fixed = fixed;
}

bool Cell::isBoundary() const {
    return m_isBoundary;
}

void Cell::setBoundary(bool isBoundary) {
    m_isBoundary = isBoundary;
    if (isBoundary) {
        setType(CellType::BOUNDARY);
    } else {
        if (m_type == CellType::BOUNDARY) setType(CellType::FLUID);
    }
}

bool Cell::isObstacle() const {
    return m_isObstacle;
}

void Cell::setObstacle(bool isObstacle) {
    m_isObstacle = isObstacle;
    if (isObstacle) {
        setType(CellType::SOLID);
    } else {
        if (m_type == CellType::SOLID) setType(CellType::FLUID);
    }
}

bool Cell::getFlag(CellFlag flag) const {
    if (flag == CellFlag::COUNT)
        return false;
    return (m_flags & static_cast<uint64_t>(flag)) != 0;
}

void Cell::setFlag(CellFlag flag, bool value) {
    if (flag == CellFlag::COUNT)
        return;
    if (value)
        m_flags |= static_cast<uint64_t>(flag);
    else
        m_flags &= ~static_cast<uint64_t>(flag);
}

//------------------------------------------------------------------------------
// Cell-Centered Physical Property Accessors (SI Units)
//------------------------------------------------------------------------------
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

double Cell::getVelocityU() const { return m_velocityU; }
void Cell::setVelocityU(double vx) { m_velocityU = vx; }

double Cell::getVelocityV() const { return m_velocityV; }
void Cell::setVelocityV(double vy) { m_velocityV = vy; }

//------------------------------------------------------------------------------
// Unit Conversion Methods
//------------------------------------------------------------------------------
void Cell::setTemperatureWithUnits(double temperature, const std::string &unit) {
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
    const Vertex &v = m_vertices[static_cast<size_t>(pos)];
    return {v.vx, v.vy};
}

void Cell::setVertexVelocity(VertexPosition pos, double vx, double vy) {
    m_vertices[static_cast<size_t>(pos)].vx = vx;
    m_vertices[static_cast<size_t>(pos)].vy = vy;
}

void Cell::setVertexVelocityWithUnits(VertexPosition pos, double vx, double vy, const std::string &unit) {
    double vx_mps = Units::convert(vx, unit, "m/s");
    double vy_mps = Units::convert(vy, unit, "m/s");
    setVertexVelocity(pos, vx_mps, vy_mps);
}

std::pair<double, double> Cell::getVertexVelocityWithUnits(VertexPosition pos, const std::string &unit) const {
    auto [vx, vy] = getVertexVelocity(pos);
    return {Units::convert(vx, "m/s", unit), Units::convert(vy, "m/s", unit)};
}

Cell::Vertex& Cell::getVertex(VertexPosition pos) {
    return m_vertices[static_cast<size_t>(pos)];
}

const Cell::Vertex& Cell::getVertex(VertexPosition pos) const {
    return m_vertices[static_cast<size_t>(pos)];
}

//------------------------------------------------------------------------------
// Material Integration
//------------------------------------------------------------------------------
std::shared_ptr<Material> Cell::getMaterial() const {
    return m_material;
}

void Cell::setMaterial(std::shared_ptr<Material> material) {
    m_material = material;
}

void Cell::mixMaterial(const std::shared_ptr<Material> &other, double mixFraction, const std::string &mixingRule) {
    if (m_material) {
        m_material = m_material->createMixture(other, mixFraction, mixingRule);
    } else {
        m_material = other;
    }
}

double Cell::getEffectiveDensity() const {
    double T = getTemperature();
    if (m_material) {
        return m_material->getPropertyAtTemperature(Material::MaterialProperty::DENSITY, T);
    }
    return getDensity();
}

//------------------------------------------------------------------------------
// Fixed Property Access (for PropertyType)
void Cell::setProperty(PropertyType type, double value) {
    m_fixedProperties[static_cast<size_t>(type)] = value;
}

double Cell::getProperty(PropertyType type) const {
    return m_fixedProperties[static_cast<size_t>(type)];
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
    m_dynamicProperties[name] = value;
}

double Cell::getDynamicProperty(const std::string &name, double defaultValue) const {
    auto it = m_dynamicProperties.find(name);
    return (it != m_dynamicProperties.end()) ? it->second : defaultValue;
}

//------------------------------------------------------------------------------
// Reset: resets cell state to default values.
void Cell::reset() {
    // Preserve type, boundary and obstacle flags.
    CellType type = m_type;
    bool isBoundary = m_isBoundary;
    bool isObstacle = m_isObstacle;
    
    // Reset physical properties to defaults.
    m_temperature = 293.15;
    m_pressure = 101325.0;
    m_density = 1.0;
    m_velocityU = 0.0;
    m_velocityV = 0.0;
    
    // Reset vertex velocities.
    for (auto &v : m_vertices) {
        v.vx = 0.0;
        v.vy = 0.0;
    }
    
    // Reset fixed properties.
    m_fixedProperties.fill(0.0);
    
    // Clear dynamic properties.
    m_dynamicProperties.clear();
    
    // Restore type and flags.
    m_type = type;
    m_isBoundary = isBoundary;
    m_isObstacle = isObstacle;
    m_flags = 0;
    if (m_isBoundary)
        m_flags |= static_cast<uint64_t>(CellFlag::IS_BOUNDARY);
    if (m_isObstacle)
        m_flags |= static_cast<uint64_t>(CellFlag::IS_OBSTACLE);
    
    // Note: Material is preserved.
}

//------------------------------------------------------------------------------
// Serialization
//------------------------------------------------------------------------------
std::string Cell::serialize() const {
    std::ostringstream oss;
    oss << "CELL_V1\n";
    oss << "TYPE=" << static_cast<int>(m_type) << "\n";
    oss << "FIXED=" << (m_fixed ? 1 : 0) << "\n";
    oss << "IS_BOUNDARY=" << (m_isBoundary ? 1 : 0) << "\n";
    oss << "IS_OBSTACLE=" << (m_isObstacle ? 1 : 0) << "\n";
    oss << "FLAGS=" << m_flags << "\n";
    oss << "TEMP=" << m_temperature << "\n";
    oss << "PRES=" << m_pressure << "\n";
    oss << "DENS=" << m_density << "\n";
    oss << "VEL_U=" << m_velocityU << "\n";
    oss << "VEL_V=" << m_velocityV << "\n";
    // For simplicity, we store material name.
    oss << "MAT=" << (m_material ? m_material->getName() : "") << "\n";
    // Serialize vertex velocities.
    oss << "VERTICES=";
    for (size_t i = 0; i < m_vertices.size(); ++i) {
        if (i > 0) oss << ",";
        oss << m_vertices[i].vx << ";" << m_vertices[i].vy;
    }
    oss << "\n";
    // Serialize fixed properties.
    oss << "PROPS=";
    for (size_t i = 0; i < m_fixedProperties.size(); ++i) {
        if (i > 0) oss << ",";
        oss << m_fixedProperties[i];
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
        std::string value = line.substr(pos+1);
        if (key == "TYPE") {
            setType(static_cast<CellType>(std::stoi(value)));
        } else if (key == "FIXED") {
            setFixed(std::stoi(value) != 0);
        } else if (key == "IS_BOUNDARY") {
            m_isBoundary = (std::stoi(value) != 0);
        } else if (key == "IS_OBSTACLE") {
            m_isObstacle = (std::stoi(value) != 0);
        } else if (key == "FLAGS") {
            m_flags = std::stoull(value);
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
        } else if (key == "VERTICES") {
            std::istringstream viss(value);
            std::string component;
            size_t idx = 0;
            while (std::getline(viss, component, ',') && idx < m_vertices.size()) {
                size_t semicolonPos = component.find(';');
                if (semicolonPos != std::string::npos) {
                    m_vertices[idx].vx = std::stod(component.substr(0, semicolonPos));
                    m_vertices[idx].vy = std::stod(component.substr(semicolonPos+1));
                }
                idx++;
            }
        } else if (key == "PROPS") {
            std::istringstream pss(value);
            std::string propVal;
            size_t idx = 0;
            while (std::getline(pss, propVal, ',') && idx < m_fixedProperties.size()) {
                m_fixedProperties[idx++] = std::stod(propVal);
            }
        } else if (key == "DYNPROPS") {
            int count = std::stoi(value);
            m_dynamicProperties.clear();
            for (int i = 0; i < count && std::getline(iss, line); ++i) {
                size_t eqPos = line.find('=');
                if (eqPos != std::string::npos) {
                    m_dynamicProperties[line.substr(0, eqPos)] = std::stod(line.substr(eqPos+1));
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
    oss << "Cell [Type: " << cellTypeToString(m_type) << "]\n";
    oss << "  Fixed: " << (m_fixed ? "Yes" : "No") << "\n";
    oss << "  Temperature: " << m_temperature << " K (" << getTemperatureWithUnits("C") << " °C)\n";
    oss << "  Pressure: " << m_pressure << " Pa (" << getPressureWithUnits("bar") << " bar)\n";
    oss << "  Density: " << m_density << " kg/m³\n";
    oss << "  Velocity: [" << m_velocityU << ", " << m_velocityV << "] m/s\n";
    oss << "  Physical Center: " << getPhysicalCenterString() << "\n";
    if (m_material)
        oss << "  Material: " << m_material->getName() << "\n";
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
    double x = (getPhysicalX() - cellSize/2) * scale;
    double y = (getPhysicalY() - cellSize/2) * scale;
    double size = cellSize * scale;
    svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << size+10 << "\" height=\"" << size+10 << "\">\n";
    svg << "  <rect x=\"" << x << "\" y=\"" << y << "\" width=\"" << size << "\" height=\"" << size
        << "\" fill=\"rgb(200,230,255)\" stroke=\"black\" stroke-width=\"1\" />\n";
    if (showVelocity) {
        double mag = std::sqrt(m_velocityU * m_velocityU + m_velocityV * m_velocityV);
        if (mag > 1e-6) {
            double cx = getPhysicalX() * scale;
            double cy = getPhysicalY() * scale;
            double vx = m_velocityU / mag;
            double vy = m_velocityV / mag;
            double arrowLen = 3.0 * scale;
            double ex = cx + vx * arrowLen;
            double ey = cy + vy * arrowLen;
            svg << "  <line x1=\"" << cx << "\" y1=\"" << cy << "\" x2=\"" << ex << "\" y2=\"" << ey
                << "\" stroke=\"black\" stroke-width=\"2\" />\n";
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
        case CellType::FLUID: return "FLUID";
        case CellType::SOLID: return "SOLID";
        case CellType::BOUNDARY: return "BOUNDARY";
        default: return "UNKNOWN";
    }
}

std::string Cell::cellFlagToString(CellFlag flag) {
    switch (flag) {
        case CellFlag::IS_INLET: return "INLET";
        case CellFlag::IS_OUTLET: return "OUTLET";
        case CellFlag::IS_WALL: return "WALL";
        case CellFlag::IS_SYMMETRY: return "SYMMETRY";
        case CellFlag::IS_BOUNDARY: return "BOUNDARY";
        case CellFlag::IS_OBSTACLE: return "OBSTACLE";
        default: return "UNKNOWN";
    }
}


