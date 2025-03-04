/**
 * ====================================================================
 * Material Class - CFD Implementation Documentation
 * ====================================================================
 *
 * The Material class defines the physical properties of substances within the
 * CFD simulation. It manages temperature-dependent property models, provides
 * unit conversion, and supports material mixtures for multi-component flows.
 *
 * Relationship with Cell and Grid:
 * -------------------------------
 * - Cell Dependency:
 *   Each Cell object contains a shared_ptr<Material> that defines the physical
 *   behavior of the fluid/solid at that location. The Material provides
 *   essential properties like density, viscosity, and thermal conductivity that
 *   determine how cells evolve during simulation.
 *
 * - Grid Integration:
 *   While not directly referenced by Grid, Materials influence the entire
 *   simulation domain through their association with Cell objects. Materials
 *   are critical for fluid-structure interaction and multi-material simulations
 *   since different regions of the Grid may contain different materials.
 *
 * Primary Use Cases:
 * ----------------
 * 1. Physical Property Management
 *    - Provides temperature-dependent material properties for flow calculations
 *    - Enforces physical consistency through appropriate unit conversions
 *    - Supports predefined materials (water, air, various metals, oils, gases)
 *
 * 2. Multi-physics Simulation
 *    - Enables heat transfer calculations through thermal properties
 *    - Supports variable-property flows where density and viscosity change with
 *      temperature
 *    - Allows modeling of complex fluid behavior through customizable property
 *      models
 *
 * 3. Material Mixing and Interfaces
 *    - Creates composite materials through mixture functionality
 *    - Supports different mixing rules (linear, logarithmic, harmonic) for
 *      physical accuracy
 *    - Enables modeling of multi-component flows with material gradients
 *
 * 4. Property Model Framework
 *    - Implements multiple temperature dependence models (constant, linear,
 *      polynomial, exponential)
 *    - Allows custom property functions for specialized material behavior
 *    - Maintains reference temperatures for consistent property calculations
 *
 * Key Material Properties:
 * ---------------------
 * - Transport Properties: Dynamic viscosity, thermal conductivity
 * - Thermodynamic Properties: Density, specific heat
 * - Additional Properties: Thermal expansion, surface tension, electrical
 * conductivity
 *
 * Material Types:
 * ------------
 *    FLUID: Liquids and gases with flow properties (viscosity)
 *    SOLID: Non-flowing materials with thermal properties
 *    INTERFACE: Special properties for fluid interfaces
 *
 * Structure Diagram:
 * ----------------
 *    +---------------+            +-----------------+
 *    | Cell          |            | Material        |
 *    |---------------| references |-----------------|
 *    | m_material    |----------->| m_type          |
 *    | m_temperature |            | m_baseProperties|
 *    | m_pressure    |            | m_propertyModels|
 *    | m_density     |            | m_customFunctions
 *    +---------------+            +-----------------+
 *                                        ^
 *                                        |
 *                                 +------+------+
 *                                 | Mixture     |
 *                                 | Components  |
 *                                 +-------------+
 *
 * Performance Considerations:
 * -------------------------
 * - Property calculations add computational overhead, especially with complex
 *   models
 * - Material mixing operations can be expensive for large numbers of components
 * - Consider caching property values for fixed temperature ranges
 * - Custom property functions should be optimized for frequently accessed
 *   properties
 *
 * Implementation Notes:
 * ------------------
 * - Material objects are typically shared between multiple cells to save memory
 * - Predefined materials include reference values and appropriate temperature
 *   models
 * - Unit conversion is handled transparently through the Units helper class
 * - Property values maintain SI units internally (kg/m³, Pa·s, W/(m·K),
 *   J/(kg·K))
 *
 * LIMITATIONS:
 * -----------
 * 1. CURRENT IMPLEMENTATION:
 *    - Limited to single-phase materials (no explicit phase change modeling)
 *    - Assumes smooth property transitions with temperature (no
 *      discontinuities)
 *    - Pressure dependence of properties not directly supported
 *
 * 2. MIXING FUNCTIONALITY:
 *    - Simple mixing rules may not capture non-linear interactions between
 *      components
 *    - No automatic handling of chemical reactions between mixed materials
 *    - Memory overhead increases with mixture complexity
 *
 * 3. COMPUTATIONAL EFFICIENCY:
 *    - Temperature-dependent properties require recalculation at each
 *      simulation step
 *    - Complex property models (polynomial, custom functions) increase CPU time
 *    - No built-in property interpolation tables for faster lookup
 *
 * 4. FRAMEWORK CONSTRAINTS:
 *    - New material properties require enum modification and rebuild
 *    - Shared pointer ownership must be carefully managed to prevent leaks
 *    - No built-in validation that material type matches expected cell behavior
 */
#pragma once

#include "Units.h"

#include <string>
#include <memory>
#include <vector>
#include <functional>
#include <array>
#include <map>

/**
 * @class Material
 * @brief Defines physical properties of materials in the CFD simulation
 * 
 * This class provides physical properties of materials (fluids or solids)
 * with support for temperature-dependent properties and material mixing.
 */
class Material {
public:
  /**
   * @enum MaterialType
   * @brief Defines the general type of material
   */
  enum class MaterialType {
    FLUID,      ///< Liquid or gas
    SOLID,      ///< Solid material
    INTERFACE   ///< Special material for fluid interfaces
  };

  /**
   * @enum PropertyModel
   * @brief Defines how a property varies with temperature
   */
  enum class PropertyModel {
    CONSTANT,           ///< Property does not change with temperature
    LINEAR,             ///< Property varies linearly with temperature
    POLYNOMIAL,         ///< Property follows a polynomial function of temperature
    EXPONENTIAL,        ///< Property follows an exponential function of temperature
    CUSTOM             ///< Property follows a custom function
  };

  /**
   * @enum MaterialProperty
   * @brief Defines the various physical properties of materials
   */
  enum class MaterialProperty {
    DENSITY,                ///< Density (kg/m³)
    DYNAMIC_VISCOSITY,      ///< Dynamic viscosity (Pa·s)
    THERMAL_CONDUCTIVITY,   ///< Thermal conductivity (W/(m·K))
    SPECIFIC_HEAT,          ///< Specific heat capacity (J/(kg·K))
    THERMAL_EXPANSION,      ///< Thermal expansion coefficient (1/K)
    SURFACE_TENSION,        ///< Surface tension (N/m) - for fluid interfaces
    ELECTRICAL_CONDUCTIVITY,///< Electrical conductivity (S/m)
    COUNT                  ///< Keep last - number of properties
  };

  /**
   * @brief Default constructor for Material
   */
  Material();
    
  /**
   * @brief Constructor with material type and name
   * @param type The type of material (fluid, solid, interface)
   * @param name The name of the material
   */
  Material(MaterialType type, const std::string& name);
    
  /**
   * @brief Get the material type
   * @return The material type
   */
  MaterialType getType() const;
    
  /**
   * @brief Get the material name
   * @return The material name
   */
  const std::string& getName() const;

  /**
   * @brief Set a base property value (at reference temperature)
   * @param property The property to set
   * @param value The property value
   */
  void setProperty(MaterialProperty property, double value);
    
  /**
   * @brief Get a base property value (at reference temperature)
   * @param property The property to get
   * @return The property value
   */
  double getProperty(MaterialProperty property) const;
    
  /**
   * @brief Set the property temperature model
   * @param property The property to set the model for
   * @param model The temperature dependence model
   * @param coefficients The coefficients for the model
   */
  void setPropertyModel(MaterialProperty property, PropertyModel model, 
			const std::vector<double>& coefficients = {});
    
  /**
   * @brief Set a custom property function
   * @param property The property to set the function for
   * @param function The custom function taking temperature and returning property value
   */
  void setCustomPropertyFunction(MaterialProperty property, 
				 std::function<double(double)> function);
    
  /**
   * @brief Get property value at specific temperature
   * @param property The property to get
   * @param temperature The temperature (K)
   * @return The property value at the given temperature
   */
  double getPropertyAtTemperature(MaterialProperty property, double temperature) const;
    
  /**
   * @brief Set the reference temperature
   * @param temperature The reference temperature (K)
   */
  void setReferenceTemperature(double temperature);
    
  /**
   * @brief Get the reference temperature
   * @return The reference temperature (K)
   */
  double getReferenceTemperature() const;
    
  /**
   * @brief Create a mixture of two materials
   * @param other The material to mix with
   * @param mixFraction The fraction of the other material (0.0 to 1.0)
   * @param mixingRule The mixing rule to use (linear, logarithmic, etc.)
   * @return A new material representing the mixture
   */
  std::shared_ptr<Material>
  createMixture(std::shared_ptr<Material> other, double mixFraction,
                const std::string &mixingRule = "linear") const;
  /**
   * @brief Check if the material is a mixture
   * @return True if the material is a mixture
   */
  bool isMixture() const;
    
  /**
   * @brief For a mixture material, get the components
   * @return Pairs of (component material, fraction)
   */
  const std::vector<std::pair<std::shared_ptr<Material>, double>>& getMixtureComponents() const;
    
  /**
   * @brief Create a predefined material by name
   * @param materialName The name of the predefined material (e.g., "water", "air")
   * @return A shared pointer to the created material
   */
  static std::shared_ptr<Material> createPredefined(const std::string& materialName);
    
  /**
   * @brief Get a human-readable name for a property
   * @param property The property
   * @return The property name
   */
  static const std::string& getPropertyName(MaterialProperty property);
    
  /**
   * @brief Get a human-readable name for a property model
   * @param model The property model
   * @return The model name
   */
  static const std::string& getModelName(PropertyModel model);

  /**
   * @brief Set a property value with unit conversion
   * @param property The property to set
   * @param value The property value in specified units
   * @param unitStr The unit string (e.g., "kg/m³", "Pa·s", "W/(m·K)")
   */
  void setPropertyWithUnits(MaterialProperty property, double value,
                            const std::string &unitStr);

  /**
   * @brief Get property value with unit conversion
   * @param property The property to get
   * @param unitStr The unit string to convert to
   * @return The property value in requested units
   */
  double getPropertyWithUnits(MaterialProperty property,
                              const std::string &unitStr) const;

  /**
   * @brief Enable or disable temperature-dependent properties
   * @param enable True to enable temperature-dependent properties, false to disable
   * 
   * When temperature-dependent properties are enabled, the material will use
   * the configured property models to calculate property values based on temperature.
   * When disabled, it will always return the base property values regardless of temperature.
   */
  void setUseTempDependentProps(bool enable) {
    m_useTempDependentProps = enable;
  }

  /**
   * @brief Set reference temperature with unit conversion
   * @param temperature Reference temperature in specified units
   * @param unitStr The temperature unit string (e.g., "K", "C", "F")
   */
  void setReferenceTemperatureWithUnits(double temperature,
                                        const std::string &unitStr) {
    double kelvin = Units::convert(temperature, unitStr, "K");
    setReferenceTemperature(kelvin);
  }

  /**
   * @brief Get reference temperature with unit conversion
   * @param unitStr The temperature unit string to convert to
   * @return Reference temperature in requested units
   */
  double getReferenceTemperatureWithUnits(const std::string &unitStr) const {
    return Units::convert(getReferenceTemperature(), "K", unitStr);
  }

  /**
   * @brief Get property value at specific temperature with unit conversion
   * @param property The property to get
   * @param temperature Temperature value in specified temperature units
   * @param temperatureUnitStr The temperature unit string (e.g., "K", "C", "F")
   * @param propertyUnitStr The property unit string for the return value
   * @return The property value at the given temperature in requested units
   */
  double
  getPropertyAtTemperatureWithUnits(MaterialProperty property,
                                    double temperature,
                                    const std::string &temperatureUnitStr,
                                    const std::string &propertyUnitStr) const;

  /**
   * @brief Create a material with properties in specified units
   * @param type The type of material
   * @param name The name of the material
   * @param density Density value
   * @param densityUnit Density unit string
   * @param dynamicViscosity Dynamic viscosity value
   * @param viscosityUnit Viscosity unit string
   * @param thermalConductivity Thermal conductivity value
   * @param conductivityUnit Thermal conductivity unit string
   * @param specificHeat Specific heat value
   * @param specificHeatUnit Specific heat unit string
   * @param refTemperature Reference temperature value
   * @param tempUnit Temperature unit string
   * @return A shared pointer to the created material
   */
  static std::shared_ptr<Material>
  createWithUnits(MaterialType type, const std::string &name, double density,
                  const std::string &densityUnit, double dynamicViscosity,
                  const std::string &viscosityUnit, double thermalConductivity,
                  const std::string &conductivityUnit, double specificHeat,
                  const std::string &specificHeatUnit,
                  double refTemperature = 293.15,
                  const std::string &tempUnit = "K");

  /**
   * @brief Get the thermal conductivity of the material at the current
   * reference temperature
   * @return Thermal conductivity value in W/(m·K)
   */
  double getThermalConductivity() const;

  /**
   * @brief Get the thermal conductivity of the material at a specific
   * temperature
   * @param temperature The temperature at which to calculate conductivity (K)
   * @return Thermal conductivity value in W/(m·K)
   */
  double getThermalConductivity(double temperature) const;

private:
  MaterialType m_type = MaterialType::FLUID;
  std::string m_name = "DefaultMaterial";
  double m_referenceTemperature = 293.15;  // Default reference temperature (20°C)
    
  // Base property values at reference temperature
  std::array<double, static_cast<size_t>(MaterialProperty::COUNT)> m_baseProperties = {};
    
  // Property models for temperature dependence
  bool m_useTempDependentProps = false;
  std::array<PropertyModel, static_cast<size_t>(MaterialProperty::COUNT)> m_propertyModels = {};
    
  // Coefficients for temperature models
  std::array<std::vector<double>, static_cast<size_t>(MaterialProperty::COUNT)> m_modelCoefficients = {};
    
  // Custom property functions
  std::array<std::function<double(double)>, static_cast<size_t>(MaterialProperty::COUNT)> m_customFunctions = {};
    
  // For mixture materials, store components and fractions
  bool m_isMixture = false;
  std::vector<std::pair<std::shared_ptr<Material>, double>> m_mixtureComponents;
    
  // Calculate property value using the specified model
  double calculatePropertyValue(MaterialProperty property, double temperature) const;
    
  // Mix properties according to mixing rules
  static double mixProperties(MaterialProperty property, double value1,
                              double value2, double fraction,
                              const std::string &rule);

  static void addScaledComponentsToMixture(
      std::shared_ptr<Material> mixture,
      const std::vector<std::pair<std::shared_ptr<Material>, double>>
          &components,
      double scaleFactor);

  void calculateMixedProperties(std::shared_ptr<Material> mixture,
                                std::shared_ptr<Material> other,
                                double mixFraction,
                                const std::string &mixingRule) const;
};
