#pragma once

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
  std::shared_ptr<Material> createMixture(std::shared_ptr<Material> other, 
					  double mixFraction, 
					  const std::string& mixingRule = "linear") const;

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

private:
  MaterialType m_type = MaterialType::FLUID;
  std::string m_name = "DefaultMaterial";
  double m_referenceTemperature = 293.15;  // Default reference temperature (20°C)
    
  // Base property values at reference temperature
  std::array<double, static_cast<size_t>(MaterialProperty::COUNT)> m_baseProperties = {};
    
  // Property models for temperature dependence
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
  static double mixProperties(MaterialProperty property, double value1, double value2, 
			      double fraction, const std::string& rule);
};


