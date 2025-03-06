/**
 * ====================================================================
 * Material Class - CFD Implementation with Optimized Data Structures
 * ====================================================================
 *
 * This version of the Material class uses flattened arrays for better
 * performance on modern computer architectures. The data structures
 * are designed for optimal cache usage and vectorization potential.
 */
#pragma once

#include "Units.h"

#include <array>
#include <functional>
#include <memory>
#include <string>
#include <vector>

/**
 * @class Material
 * @brief Defines physical properties of materials with optimized data layout
 */
class Material {
public:
  /**
   * @enum MaterialType
   * @brief Defines the general type of material
   */
  enum class MaterialType {
    FLUID,    ///< Liquid or gas
    SOLID,    ///< Solid material
    INTERFACE ///< Special material for fluid interfaces
  };

  /**
   * @enum PropertyModel
   * @brief Defines how a property varies with temperature
   */
  enum class PropertyModel {
    CONSTANT,    ///< Property does not change with temperature
    LINEAR,      ///< Property varies linearly with temperature
    POLYNOMIAL,  ///< Property follows a polynomial function of temperature
    EXPONENTIAL, ///< Property follows an exponential function of temperature
    CUSTOM       ///< Property follows a custom function
  };

  /**
   * @enum MaterialProperty
   * @brief Defines the various physical properties of materials
   */
  enum class MaterialProperty {
    DENSITY,                 ///< Density (kg/m³)
    DYNAMIC_VISCOSITY,       ///< Dynamic viscosity (Pa·s)
    THERMAL_CONDUCTIVITY,    ///< Thermal conductivity (W/(m·K))
    SPECIFIC_HEAT,           ///< Specific heat capacity (J/(kg·K))
    THERMAL_EXPANSION,       ///< Thermal expansion coefficient (1/K)
    SURFACE_TENSION,         ///< Surface tension (N/m) - for fluid interfaces
    ELECTRICAL_CONDUCTIVITY, ///< Electrical conductivity (S/m)
    COUNT                    ///< Keep last - number of properties
  };

  // Maximum coefficients per property model
  static constexpr size_t MAX_COEFFICIENTS = 8;

  // Maximum number of mixture components
  static constexpr size_t MAX_MIXTURE_COMPONENTS = 16;

  Material();
  Material(MaterialType type, const std::string &name);
  Material(const Material &other);
  Material(Material &&other) noexcept;
  ~Material();

  Material &operator=(const Material &other);
  Material &operator=(Material &&other) noexcept;

  // Material registry access
  uint32_t getID() const { return m_materialID; }
  static Material *getByID(uint32_t id);

  /**
   * @brief Get the material type
   * @return The material type
   */
  MaterialType getType() const { return m_type; }

  /**
   * @brief Get the material name
   * @return The material name
   */
  const std::string &getName() const { return m_name; }

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
                        const std::vector<double> &coefficients = {});

  /**
   * @brief Set a custom property function
   * @param property The property to set the function for
   * @param function The custom function taking temperature and returning
   * property value
   */
  void setCustomPropertyFunction(MaterialProperty property,
                                 std::function<double(double)> function);

  /**
   * @brief Get property value at specific temperature
   * @param property The property to get
   * @param temperature The temperature (K)
   * @return The property value at the given temperature
   */
  double getPropertyAtTemperature(MaterialProperty property,
                                  double temperature) const;

  /**
   * @brief Set the reference temperature
   * @param temperature The reference temperature (K)
   */
  void setReferenceTemperature(double temperature) {
    m_referenceTemperature = temperature;
  }

  /**
   * @brief Get the reference temperature
   * @return The reference temperature (K)
   */
  double getReferenceTemperature() const { return m_referenceTemperature; }

  void setMixingRule(const std::string &rule) { m_mixingRule = rule; }
  /**
   * @brief Create a mixture of two materials
   * @param other The material to mix with
   * @param mixFraction The fraction of the other material (0.0 to 1.0)
   * @param mixingRule The mixing rule to use
   * @return A new material representing the mixture
   */
  std::shared_ptr<Material>
  createMixture(std::shared_ptr<Material> other, double mixFraction,
                const std::string &mixingRule = "linear") const;

  /**
   * @brief Create a mixture of multiple materials
   * @param materials The materials to mix
   * @param fractions The fractions of each material (should sum to 1.0)
   * @param mixingRule The mixing rule to use
   * @return A new material representing the mixture
   */
  static std::shared_ptr<Material>
  createMixture(const std::vector<std::shared_ptr<Material>> &materials,
                const std::vector<double> &fractions,
                const std::string &mixingRule);

  /**
   * @brief Check if the material is a mixture
   * @return True if the material is a mixture
   */
  bool isMixture() const { return m_componentCount > 0; }

  /**
   * @brief For a mixture material, get the components
   * @return Vector of component materials and fractions
   */
  std::vector<std::pair<std::shared_ptr<Material>, double>>
  getMixtureComponents() const;

  /**
   * @brief Create a predefined material by name
   * @param materialName The name of the predefined material (e.g., "water",
   * "air")
   * @return A shared pointer to the created material
   */
  static std::shared_ptr<Material>
  createPredefined(const std::string &materialName);

  /**
   * @brief Get a human-readable name for a property
   * @param property The property
   * @return The property name
   */
  static const std::string &getPropertyName(MaterialProperty property);

  /**
   * @brief Get a human-readable name for a property model
   * @param model The property model
   * @return The model name
   */
  static const std::string &getModelName(PropertyModel model);

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
   * @param enable True to enable temperature-dependent properties, false to
   * disable
   */
  void setUseTempDependentProps(bool enable) {
    m_useTempDependentProps = enable;
  }

  bool getUsesTempDependentProps() const { return m_useTempDependentProps; }

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
   * @brief Get the thermal conductivity at the reference temperature
   * @return Thermal conductivity value in W/(m·K)
   */
  double getThermalConductivity() const {
    return getProperty(MaterialProperty::THERMAL_CONDUCTIVITY);
  }

  /**
   * @brief Get the thermal conductivity at a specific temperature
   * @param temperature The temperature at which to calculate conductivity (K)
   * @return Thermal conductivity value in W/(m·K)
   */
  double getThermalConductivity(double temperature) const {
    return getPropertyAtTemperature(MaterialProperty::THERMAL_CONDUCTIVITY,
                                    temperature);
  }

  bool isUsingTempDependentProps() const { return m_useTempDependentProps; }

  /**
   * @brief Mix properties according to mixing rules
   * @param property The property to mix
   * @param value1 First property value
   * @param value2 Second property value
   * @param fraction Fraction of second value (0-1)
   * @param rule Mixing rule name
   * @return Mixed property value
   */
  static double mixProperties(MaterialProperty property, double value1,
                              double value2, double fraction,
                              const std::string &rule);

private:
  static std::unordered_map<uint32_t, Material *> s_materialRegistry;
  std::string m_mixingRule; // User-specified mixing rule for this mixture

  // Basic material info
  MaterialType m_type;
  std::string m_name;
  double m_referenceTemperature;
  bool m_useTempDependentProps;

  // Flat array for property storage - contiguous memory layout
  alignas(
      32) double m_baseProperties[static_cast<size_t>(MaterialProperty::COUNT)];

  // Property models - one per property
  PropertyModel m_propertyModels[static_cast<size_t>(MaterialProperty::COUNT)];

  // Coefficient storage - fixed size for deterministic memory layout
  // First dimension: property index
  // Second dimension: coefficient index (up to MAX_COEFFICIENTS)
  alignas(32) double m_coefficients[static_cast<size_t>(
      MaterialProperty::COUNT)][MAX_COEFFICIENTS];

  // Number of coefficients for each property
  uint8_t m_coefficientCounts[static_cast<size_t>(MaterialProperty::COUNT)];

  // Custom functions - CPU only
  std::function<double(double)>
      m_customFunctions[static_cast<size_t>(MaterialProperty::COUNT)];

  // Mixture components with fixed-size arrays
  uint32_t m_componentIDs[MAX_MIXTURE_COMPONENTS];
  double m_componentFractions[MAX_MIXTURE_COMPONENTS];
  uint8_t m_componentCount;

  // Material registry for lookup by ID

  static uint32_t s_nextMaterialID;
  uint32_t m_materialID;

  // Register/unregister material in registry
  void registerMaterial();
  void unregisterMaterial();

  void addComponent(uint32_t materialID, double fraction);

  // Calculate property value using the specified model
  double calculatePropertyValue(MaterialProperty property,
                                double temperature) const;

  // Helper methods for material mixing
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
