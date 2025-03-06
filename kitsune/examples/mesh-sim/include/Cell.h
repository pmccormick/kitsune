/**
 * @file Cell.h
 * @brief Cell class for CFD simulations with unit‐aware property operations.
 *
 * ================================================================================
 * Design Considerations and Future Directions:
 * ================================================================================
 * This Cell class encapsulates a single computational cell. Its design preserves
 * the full interface of the original version while incorporating complete support
 * for physical unit conversions via the Units namespace. Key points include:
 *
 *   • **Internal Storage in SI Units:**
 *     All cell‐center data (temperature, pressure, density, velocity) is stored in
 *     SI units. For example, temperature is stored in Kelvin, pressure in Pascal,
 *     density in kg/m³, and velocity in m/s.
 *
 *   • **Unit‑Aware Accessors:**
 *     For each physical quantity, additional “WithUnits” methods allow users to set
 *     and get values in other units. These functions internally convert to/from SI
 *     using the Units::convert and Units::enforceValid… functions.
 *
 *   • **Full Simulation Interface:**
 *     The Cell class supports cell types (FLUID, SOLID, BOUNDARY), flags (e.g.,
 *     IS_INLET, IS_OBSTACLE), fixed state, dynamic and fixed properties (e.g., VORTICITY,
 *     STREAM_FUNCTION), and serialization. It also provides convenience methods for
 *     printing and SVG visualization.
 *
 *   • **Material Integration and Mixing:**
 *     A cell holds a shared pointer to a Material. Rather than storing multiple
 *     materials, the Cell leverages the Material class’s own mixing functionality via
 *     a mixMaterial() method. This ensures that all complex material physics remains
 *     encapsulated within the Material class.
 *
 *   • **Future Enhancements:**
 *     - In a full data‐oriented redesign, cell–center data would be stored in separate
 *       Field objects (structure‑of‑arrays) owned by a Mesh. This version, however, keeps
 *       per‑cell storage for simplicity and unit testing.
 *     - More advanced unit–error reporting, custom allocators, and GPU–specific optimizations
 *       may be added later.
 *
 * ================================================================================
 * Role in Computational Science:
 * ================================================================================
 * In simulations such as CFD, it is critical that each cell can readily convert between
 * its internal SI representation and the various units that users and external libraries
 * require. This Cell class:
 *
 *   - Bridges high-level simulation logic with low-level, unit–safe, efficient numerical
 *     computations.
 *   - Provides a clear, self-documenting interface that minimizes unit conversion errors.
 *   - Integrates with Material physics so that effective properties (like density) can be
 *     computed based on temperature-dependent or mixed–material models.
 *
 * This design is intended to be both performant and maintainable, ensuring that key physical
 * quantities are accurately represented and easily converted for diverse simulation needs.
 *
 * ================================================================================
 */

#ifndef CELL_H
#define CELL_H

#include "Units.h"
#include "Material.h"

#include <array>
#include <unordered_map>
#include <string>
#include <sstream>
#include <iostream>
#include <cstdint>
#include <algorithm>
#include <memory>

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
        IS_INLET    = 0x00000001,
        IS_OUTLET   = 0x00000002,
        IS_WALL     = 0x00000004,
        IS_SYMMETRY = 0x00000008,
        IS_BOUNDARY = 0x00000010,
        IS_OBSTACLE = 0x00000020,
        COUNT       = 0x00000040  // not used as a flag
    };

    // Vertex structure for storing vertex velocities.
    struct Vertex {
        double vx = 0.0;
        double vy = 0.0;
    };

    // Default constructor (for unit testing purposes).
    Cell();

    // Constructors with type specification.
    Cell(CellType type);

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
    void setVertexVelocityWithUnits(VertexPosition pos, double vx, double vy, const std::string &unit);
    std::pair<double, double> getVertexVelocityWithUnits(VertexPosition pos, const std::string &unit) const;
    // Also provide direct vertex access.
    Vertex& getVertex(VertexPosition pos);
    const Vertex& getVertex(VertexPosition pos) const;

    // --- Material Integration ---
    std::shared_ptr<Material> getMaterial() const;
    void setMaterial(std::shared_ptr<Material> material);
    /**
     * @brief Mixes the cell’s current material with another using the Material class’s
     *        mixing function.
     * @param other Shared pointer to the other material.
     * @param mixFraction Fraction of the other material.
     * @param mixingRule Mixing rule string (default uses Material’s default).
     */
    void mixMaterial(const std::shared_ptr<Material> &other, double mixFraction, const std::string &mixingRule = "default");

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
    double getDynamicProperty(const std::string &name, double defaultValue = 0.0) const;

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

private:
    // Basic cell state.
    CellType m_type = CellType::FLUID;
    bool m_fixed = false;
    bool m_isBoundary = false;
    bool m_isObstacle = false;
    uint64_t m_flags = 0;

    // Basic physical properties (stored in SI units).
    double m_temperature = 293.15; // Kelvin (default: 20°C)
    double m_pressure = 101325.0;  // Pascal (1 atm)
    double m_density = 1.0;        // kg/m³
    double m_velocityU = 0.0;      // m/s, x-component
    double m_velocityV = 0.0;      // m/s, y-component

    // Vertex velocities (for the four corners of the cell).
    std::array<Vertex, 4> m_vertices;

    // Fixed properties storage (one per PropertyType).
    std::array<double, static_cast<size_t>(PropertyType::COUNT)> m_fixedProperties{};

    // Dynamic properties (stored by name).
    std::unordered_map<std::string, double> m_dynamicProperties;

    // Associated material.
    std::shared_ptr<Material> m_material = nullptr;
};

#endif // CELL_H

