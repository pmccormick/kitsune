#include "Material.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

// Static property name strings
static const std::array<std::string, static_cast<size_t>(Material::MaterialProperty::COUNT)> PROPERTY_NAMES = {
  "Density", "DynamicViscosity", "ThermalConductivity", "SpecificHeat", 
  "ThermalExpansion", "SurfaceTension", "ElectricalConductivity"
};

// Static model name strings
static const std::array<std::string, 5> MODEL_NAMES = {
  "Constant", "Linear", "Polynomial", "Exponential", "Custom"
};

Material::Material() 
  : m_type(MaterialType::FLUID),
    m_name("DefaultMaterial"),
    m_referenceTemperature(293.15),
    m_isMixture(false) {
    
  // Initialize all properties to zero
  m_baseProperties.fill(0.0);
    
  // Initialize all properties to constant model
  m_propertyModels.fill(PropertyModel::CONSTANT);
}

Material::Material(MaterialType type, const std::string& name) 
  : m_type(type),
    m_name(name),
    m_referenceTemperature(293.15),
    m_isMixture(false) {
    
  // Initialize all properties to zero
  m_baseProperties.fill(0.0);
    
  // Initialize all properties to constant model
  m_propertyModels.fill(PropertyModel::CONSTANT);
}

Material::MaterialType Material::getType() const {
  return m_type;
}

const std::string& Material::getName() const {
  return m_name;
}

void Material::setProperty(MaterialProperty property, double value) {
  m_baseProperties[static_cast<size_t>(property)] = value;
}

double Material::getProperty(MaterialProperty property) const {
  return m_baseProperties[static_cast<size_t>(property)];
}

void Material::setPropertyModel(MaterialProperty property, PropertyModel model, 
				const std::vector<double>& coefficients) {
  m_propertyModels[static_cast<size_t>(property)] = model;
    
  // Only store coefficients if provided
  if (!coefficients.empty()) {
    m_modelCoefficients[static_cast<size_t>(property)] = coefficients;
  }
    
  // For CONSTANT model, ensure we always have at least one coefficient
  if (model == PropertyModel::CONSTANT && 
      m_modelCoefficients[static_cast<size_t>(property)].empty()) {
    m_modelCoefficients[static_cast<size_t>(property)].push_back(
								 m_baseProperties[static_cast<size_t>(property)]);
  }
}

void Material::setCustomPropertyFunction(MaterialProperty property, 
					 std::function<double(double)> function) {
  m_propertyModels[static_cast<size_t>(property)] = PropertyModel::CUSTOM;
  m_customFunctions[static_cast<size_t>(property)] = function;
}

double Material::calculatePropertyValue(MaterialProperty property, double temperature) const {
  size_t index = static_cast<size_t>(property);
  double baseValue = m_baseProperties[index];
  PropertyModel model = m_propertyModels[index];
    
  // If the material is a mixture, calculate each component and mix them
  if (m_isMixture) {
    double mixedValue = 0.0;
    for (const auto& component : m_mixtureComponents) {
      double componentValue = component.first->getPropertyAtTemperature(property, temperature);
      mixedValue = mixProperties(property, mixedValue, componentValue, component.second, "linear");
    }
    return mixedValue;
  }
    
  // For pure materials, calculate based on the model
  switch (model) {
  case PropertyModel::CONSTANT:
    return baseValue;
            
  case PropertyModel::LINEAR: {
    // Linear model: value = baseValue * (1 + a*(T-Tref))
    // coefficients[0] = a (linear coefficient)
    const auto& coeffs = m_modelCoefficients[index];
    if (coeffs.empty()) return baseValue; // Fallback to constant if no coefficients
            
    double deltaT = temperature - m_referenceTemperature;
    return baseValue * (1.0 + coeffs[0] * deltaT);
  }
            
  case PropertyModel::POLYNOMIAL: {
    // Polynomial model: value = baseValue * (1 + a1*dT + a2*dT^2 + a3*dT^3 + ...)
    // coefficients = [a1, a2, a3, ...]
    const auto& coeffs = m_modelCoefficients[index];
    if (coeffs.empty()) return baseValue; // Fallback to constant if no coefficients
            
    double deltaT = temperature - m_referenceTemperature;
    double factor = 1.0;
            
    for (size_t i = 0; i < coeffs.size(); ++i) {
      factor += coeffs[i] * std::pow(deltaT, i + 1);
    }
            
    return baseValue * factor;
  }
            
  case PropertyModel::EXPONENTIAL: {
    // Exponential model: value = baseValue * exp(a*(T-Tref))
    // coefficients[0] = a (exponential coefficient)
    const auto& coeffs = m_modelCoefficients[index];
    if (coeffs.empty()) return baseValue; // Fallback to constant if no coefficients
            
    double deltaT = temperature - m_referenceTemperature;
    return baseValue * std::exp(coeffs[0] * deltaT);
  }
            
  case PropertyModel::CUSTOM: {
    // Custom function model
    if (m_customFunctions[index]) {
      return m_customFunctions[index](temperature);
    }
    return baseValue; // Fallback if no function is set
  }
            
  default:
    return baseValue;
  }
}

double Material::getPropertyAtTemperature(MaterialProperty property, double temperature) const {
  return calculatePropertyValue(property, temperature);
}

void Material::setReferenceTemperature(double temperature) {
  m_referenceTemperature = temperature;
}

double Material::getReferenceTemperature() const {
  return m_referenceTemperature;
}

// Updates to Material.cpp - Replace the existing createMixture implementation

std::shared_ptr<Material>
Material::createMixture(std::shared_ptr<Material> other, double mixFraction,
                        const std::string &mixingRule) const {
  // Ensure the mix fraction is in valid range
  mixFraction = std::max(0.0, std::min(1.0, mixFraction));

  // Create a new material for the mixture
  auto mixture = std::make_shared<Material>(
      MaterialType::FLUID, m_name + "-" + other->getName() + "-Mixture");

  // Mark as mixture
  mixture->m_isMixture = true;

  // Reserve space for components to avoid reallocations
  mixture->m_mixtureComponents.reserve(
      (m_isMixture ? m_mixtureComponents.size() : 1) +
      (other->m_isMixture ? other->m_mixtureComponents.size() : 1));

  // Add components efficiently based on mixture status
  if (m_isMixture && other->m_isMixture) {
    // Both are mixtures - scale and add components
    addScaledComponentsToMixture(mixture, m_mixtureComponents,
                                 1.0 - mixFraction);
    addScaledComponentsToMixture(mixture, other->m_mixtureComponents,
                                 mixFraction);
  } else if (m_isMixture) {
    // First is mixture, second is pure
    addScaledComponentsToMixture(mixture, m_mixtureComponents,
                                 1.0 - mixFraction);
    mixture->m_mixtureComponents.emplace_back(other, mixFraction);
  } else if (other->m_isMixture) {
    // First is pure, second is mixture
    mixture->m_mixtureComponents.emplace_back(std::make_shared<Material>(*this),
                                              1.0 - mixFraction);
    addScaledComponentsToMixture(mixture, other->m_mixtureComponents,
                                 mixFraction);
  } else {
    // Both materials are pure
    mixture->m_mixtureComponents.emplace_back(std::make_shared<Material>(*this),
                                              1.0 - mixFraction);
    mixture->m_mixtureComponents.emplace_back(other, mixFraction);
  }

  // Efficiently calculate and set properties
  calculateMixedProperties(mixture, other, mixFraction, mixingRule);

  return mixture;
}

void Material::addScaledComponentsToMixture(
    std::shared_ptr<Material> mixture,
    const std::vector<std::pair<std::shared_ptr<Material>, double>> &components,
    double scaleFactor) {

  for (const auto &component : components) {
    mixture->m_mixtureComponents.emplace_back(component.first,
                                              component.second * scaleFactor);
  }
}

void Material::calculateMixedProperties(std::shared_ptr<Material> mixture,
                                        std::shared_ptr<Material> other,
                                        double mixFraction,
                                        const std::string &mixingRule) const {

  // Pre-calculate all properties at once to minimize temperature lookups
  std::array<double, static_cast<size_t>(MaterialProperty::COUNT)> thisProps;
  std::array<double, static_cast<size_t>(MaterialProperty::COUNT)> otherProps;

  // Calculate all properties for each material (avoids repeated temperature
  // lookups)
  for (size_t i = 0; i < static_cast<size_t>(MaterialProperty::COUNT); ++i) {
    MaterialProperty prop = static_cast<MaterialProperty>(i);
    thisProps[i] = getPropertyAtTemperature(prop, m_referenceTemperature);
    otherProps[i] =
        other->getPropertyAtTemperature(prop, m_referenceTemperature);
  }

  // Mix and set properties
  for (size_t i = 0; i < static_cast<size_t>(MaterialProperty::COUNT); ++i) {
    MaterialProperty prop = static_cast<MaterialProperty>(i);
    double mixedValue = mixProperties(prop, thisProps[i], otherProps[i],
                                      mixFraction, mixingRule);
    mixture->setProperty(prop, mixedValue);
  }

  // Set reference temperature to match this material
  mixture->setReferenceTemperature(m_referenceTemperature);
}

// Add implementation of createWithUnits that was missing
std::shared_ptr<Material> Material::createWithUnits(
    MaterialType type, const std::string &name, double density,
    const std::string &densityUnit, double dynamicViscosity,
    const std::string &viscosityUnit, double thermalConductivity,
    const std::string &conductivityUnit, double specificHeat,
    const std::string &specificHeatUnit, double refTemperature,
    const std::string &tempUnit) {

  // Create the material
  auto material = std::make_shared<Material>(type, name);

  // Set reference temperature with unit conversion
  double kelvin = Units::convert(refTemperature, tempUnit, "K");
  material->setReferenceTemperature(kelvin);

  // Set properties with unit conversion
  double densitySI = Units::convert(density, densityUnit, "kg/m³");
  material->setProperty(MaterialProperty::DENSITY, densitySI);

  double viscositySI = Units::convert(dynamicViscosity, viscosityUnit, "Pa·s");
  material->setProperty(MaterialProperty::DYNAMIC_VISCOSITY, viscositySI);

  double conductivitySI =
      Units::convert(thermalConductivity, conductivityUnit, "W/(m·K)");
  material->setProperty(MaterialProperty::THERMAL_CONDUCTIVITY, conductivitySI);

  double specificHeatSI =
      Units::convert(specificHeat, specificHeatUnit, "J/(kg·K)");
  material->setProperty(MaterialProperty::SPECIFIC_HEAT, specificHeatSI);

  return material;
}

double Material::mixProperties(MaterialProperty property, double value1, double value2, 
			       double fraction, const std::string& rule) {
  // Use different mixing rules based on property and specified rule
  if (rule == "linear") {
    // Simple linear interpolation: value = (1-f)*v1 + f*v2
    return (1.0 - fraction) * value1 + fraction * value2;
  }
  else if (rule == "logarithmic") {
    // Logarithmic interpolation: ln(value) = (1-f)*ln(v1) + f*ln(v2)
    // Useful for properties like viscosity
    if (value1 <= 0.0 || value2 <= 0.0) {
      return (1.0 - fraction) * value1 + fraction * value2; // Fallback to linear if values <= 0
    }
    return std::exp((1.0 - fraction) * std::log(value1) + fraction * std::log(value2));
  }
  else if (rule == "harmonic") {
    // Harmonic mean: 1/value = (1-f)/v1 + f/v2
    // Useful for resistivity-like properties
    if (value1 == 0.0 || value2 == 0.0) {
      return 0.0; // Avoid division by zero
    }
    double invValue = (1.0 - fraction) / value1 + fraction / value2;
    return 1.0 / invValue;
  }
  else if (rule == "max") {
    // Maximum value
    return std::max(value1, value2);
  }
  else if (rule == "min") {
    // Minimum value
    return std::min(value1, value2);
  }
  else {
    // Default to linear mixing
    return (1.0 - fraction) * value1 + fraction * value2;
  }
}

bool Material::isMixture() const {
    return m_isMixture;
}

const std::vector<std::pair<std::shared_ptr<Material>, double>>& Material::getMixtureComponents() const {
    return m_mixtureComponents;
}

const std::string& Material::getPropertyName(MaterialProperty property) {
    return PROPERTY_NAMES[static_cast<size_t>(property)];
}

const std::string& Material::getModelName(PropertyModel model) {
    return MODEL_NAMES[static_cast<size_t>(model)];
}

std::shared_ptr<Material>
Material::createPredefined(const std::string &materialName) {

  if (materialName == "water") {
    return createWithUnits(MaterialType::FLUID, "Water", 998.2,
                           "kg/m³",            // Density
                           1.0016e-3, "Pa·s",  // Dynamic viscosity
                           0.6, "W/(m·K)",     // Thermal conductivity
                           4182.0, "J/(kg·K)", // Specific heat
                           293.15, "K"         // Reference temperature (20°C)
    );
  } else if (materialName == "air") {
    return createWithUnits(MaterialType::FLUID, "Air", 1.204,
                           "kg/m³",            // Density
                           1.825e-5, "Pa·s",   // Dynamic viscosity
                           0.0257, "W/(m·K)",  // Thermal conductivity
                           1005.0, "J/(kg·K)", // Specific heat
                           293.15, "K"         // Reference temperature (20°C)
    );
  }
  // Metals
  else if (materialName == "aluminum") {
    auto material = createWithUnits(MaterialType::SOLID, "Aluminum", 2700.0,
                                    "kg/m³",     // Density
                                    0.0, "Pa·s", // Viscosity (N/A for solids)
                                    237.0, "W/(m·K)",  // Thermal conductivity
                                    900.0, "J/(kg·K)", // Specific heat
                                    293.15, "K" // Reference temperature (20°C)
    );
    material->setProperty(MaterialProperty::THERMAL_EXPANSION,
                          2.31e-5); // 1/K
    material->setProperty(MaterialProperty::ELECTRICAL_CONDUCTIVITY,
                          3.5e7); // S/m

    // Metals have slight temperature dependence on thermal conductivity
    material->setPropertyModel(MaterialProperty::THERMAL_CONDUCTIVITY,
                               PropertyModel::LINEAR, {-0.0004});

    // Specific heat increases slightly with temperature
    material->setPropertyModel(MaterialProperty::SPECIFIC_HEAT,
                               PropertyModel::LINEAR, {0.0005});

    return material;
  } else if (materialName == "brass") {
    auto material = createWithUnits(MaterialType::SOLID, "Brass", 8500.0,
                                    "kg/m³",     // Density
                                    0.0, "Pa·s", // Viscosity (N/A for solids)
                                    109.0, "W/(m·K)",  // Thermal conductivity
                                    380.0, "J/(kg·K)", // Specific heat
                                    293.15, "K" // Reference temperature (20°C)
    );
    material->setProperty(MaterialProperty::THERMAL_EXPANSION, 1.9e-5); // 1/K
    material->setProperty(MaterialProperty::ELECTRICAL_CONDUCTIVITY,
                          1.5e7); // S/m

    // Temperature dependence
    material->setPropertyModel(MaterialProperty::THERMAL_CONDUCTIVITY,
                               PropertyModel::LINEAR, {-0.0002});
    material->setPropertyModel(MaterialProperty::SPECIFIC_HEAT,
                               PropertyModel::LINEAR, {0.0003});

    return material;
  } else if (materialName == "copper") {
    auto material = createWithUnits(MaterialType::SOLID, "Copper", 8960.0,
                                    "kg/m³",     // Density
                                    0.0, "Pa·s", // Viscosity (N/A for solids)
                                    401.0, "W/(m·K)",  // Thermal conductivity
                                    385.0, "J/(kg·K)", // Specific heat
                                    293.15, "K" // Reference temperature (20°C)
    );
    material->setProperty(MaterialProperty::THERMAL_EXPANSION, 1.7e-5); // 1/K
    material->setProperty(MaterialProperty::ELECTRICAL_CONDUCTIVITY,
                          5.8e7); // S/m

    // Temperature dependence
    material->setPropertyModel(MaterialProperty::THERMAL_CONDUCTIVITY,
                               PropertyModel::LINEAR, {-0.0005});
    material->setPropertyModel(MaterialProperty::SPECIFIC_HEAT,
                               PropertyModel::LINEAR, {0.0002});

    return material;
  } else if (materialName == "steel") {
    auto material = createWithUnits(MaterialType::SOLID, "Steel", 7850.0,
                                    "kg/m³",     // Density
                                    0.0, "Pa·s", // Viscosity (N/A for solids)
                                    50.2, "W/(m·K)",   // Thermal conductivity
                                    490.0, "J/(kg·K)", // Specific heat
                                    293.15, "K" // Reference temperature (20°C)
    );
    material->setProperty(MaterialProperty::THERMAL_EXPANSION, 1.2e-5); // 1/K
    material->setProperty(MaterialProperty::ELECTRICAL_CONDUCTIVITY,
                          1.0e7); // S/m

    // Temperature dependence
    material->setPropertyModel(MaterialProperty::THERMAL_CONDUCTIVITY,
                               PropertyModel::LINEAR, {0.0001});
    material->setPropertyModel(MaterialProperty::SPECIFIC_HEAT,
                               PropertyModel::LINEAR, {0.0006});

    return material;
  }

  // Oils
  else if (materialName == "machine_oil") {
    auto material = createWithUnits(MaterialType::FLUID, "Machine Oil", 900.0,
                                    "kg/m³",            // Density
                                    0.11, "Pa·s",       // Dynamic viscosity
                                    0.15, "W/(m·K)",    // Thermal conductivity
                                    1900.0, "J/(kg·K)", // Specific heat
                                    293.15, "K" // Reference temperature (20°C)
    );
    material->setProperty(MaterialProperty::THERMAL_EXPANSION, 7.0e-4); // 1/K

    // High temperature dependence of viscosity
    material->setPropertyModel(MaterialProperty::DYNAMIC_VISCOSITY,
                               PropertyModel::EXPONENTIAL, {-0.025});

    // Slight temperature dependence for density
    material->setPropertyModel(MaterialProperty::DENSITY, PropertyModel::LINEAR,
                               {-0.0007});

    return material;
  } else if (materialName == "motor_oil") {
    auto material =
        createWithUnits(MaterialType::FLUID, "Motor Oil (SAE 10W-30)", 870.0,
                        "kg/m³",            // Density
                        0.16, "Pa·s",       // Dynamic viscosity
                        0.145, "W/(m·K)",   // Thermal conductivity
                        2000.0, "J/(kg·K)", // Specific heat
                        293.15, "K"         // Reference temperature (20°C)
        );
    material->setProperty(MaterialProperty::THERMAL_EXPANSION, 6.5e-4); // 1/K

    // Strong temperature dependence of viscosity
    material->setPropertyModel(MaterialProperty::DYNAMIC_VISCOSITY,
                               PropertyModel::EXPONENTIAL, {-0.028});

    // Oil thins with temperature
    material->setPropertyModel(MaterialProperty::DENSITY, PropertyModel::LINEAR,
                               {-0.00065});

    return material;
  } else if (materialName == "hydraulic_oil") {
    auto material = createWithUnits(MaterialType::FLUID, "Hydraulic Oil", 890.0,
                                    "kg/m³",            // Density
                                    0.06, "Pa·s",       // Dynamic viscosity
                                    0.14, "W/(m·K)",    // Thermal conductivity
                                    1850.0, "J/(kg·K)", // Specific heat
                                    293.15, "K" // Reference temperature (20°C)
    );
    material->setProperty(MaterialProperty::THERMAL_EXPANSION, 7.2e-4); // 1/K

    // Temperature dependence of viscosity
    material->setPropertyModel(MaterialProperty::DYNAMIC_VISCOSITY,
                               PropertyModel::EXPONENTIAL, {-0.022});

    return material;
  }
  // Gases
  else if (materialName == "nitrogen") {
    auto material = createWithUnits(MaterialType::FLUID, "Nitrogen", 1.165,
                                    "kg/m³",           // Density at 20°C, 1 atm
                                    1.76e-5, "Pa·s",   // Dynamic viscosity
                                    0.0258, "W/(m·K)", // Thermal conductivity
                                    1040.0, "J/(kg·K)", // Specific heat
                                    293.15, "K" // Reference temperature (20°C)
    );

    // For ideal gas: density inversely proportional to temperature
    material->setCustomPropertyFunction(
        MaterialProperty::DENSITY, [material](double T) -> double {
          double rho0 = material->getProperty(MaterialProperty::DENSITY);
          double T0 = material->getReferenceTemperature();
          return rho0 * (T0 / T); // assuming constant pressure
        });

    // Viscosity increases with temperature for gases
    material->setPropertyModel(MaterialProperty::DYNAMIC_VISCOSITY,
                               PropertyModel::LINEAR, {0.00026});

    // Thermal conductivity also increases with temperature
    material->setPropertyModel(MaterialProperty::THERMAL_CONDUCTIVITY,
                               PropertyModel::LINEAR, {0.00007});

    return material;
  } else if (materialName == "hydrogen") {
    auto material = createWithUnits(
        MaterialType::FLUID, "Hydrogen", 0.0837,
        "kg/m³",             // Density at 20°C, 1 atm
        8.90e-6, "Pa·s",     // Dynamic viscosity
        0.1805, "W/(m·K)",   // Thermal conductivity - very high for a gas
        14320.0, "J/(kg·K)", // Specific heat - also very high
        293.15, "K"          // Reference temperature (20°C)
    );

    // For ideal gas: density inversely proportional to temperature
    material->setCustomPropertyFunction(
        MaterialProperty::DENSITY, [material](double T) -> double {
          double rho0 = material->getProperty(MaterialProperty::DENSITY);
          double T0 = material->getReferenceTemperature();
          return rho0 * (T0 / T); // assuming constant pressure
        });

    // Viscosity increases with temperature for gases
    material->setPropertyModel(MaterialProperty::DYNAMIC_VISCOSITY,
                               PropertyModel::LINEAR, {0.00018});

    // Thermal conductivity also increases with temperature
    material->setPropertyModel(MaterialProperty::THERMAL_CONDUCTIVITY,
                               PropertyModel::LINEAR, {0.0003});

    return material;
  } else if (materialName == "carbon_dioxide") {
    auto material =
        createWithUnits(MaterialType::FLUID, "Carbon Dioxide", 1.842,
                        "kg/m³",           // Density at 20°C, 1 atm
                        1.47e-5, "Pa·s",   // Dynamic viscosity
                        0.0166, "W/(m·K)", // Thermal conductivity
                        846.0, "J/(kg·K)", // Specific heat
                        293.15, "K"        // Reference temperature (20°C)
        );

    // For ideal gas: density inversely proportional to temperature
    material->setCustomPropertyFunction(
        MaterialProperty::DENSITY, [material](double T) -> double {
          double rho0 = material->getProperty(MaterialProperty::DENSITY);
          double T0 = material->getReferenceTemperature();
          return rho0 * (T0 / T); // assuming constant pressure
        });

    // Viscosity increases with temperature for gases
    material->setPropertyModel(MaterialProperty::DYNAMIC_VISCOSITY,
                               PropertyModel::LINEAR, {0.00024});

    // Thermal conductivity also increases with temperature
    material->setPropertyModel(MaterialProperty::THERMAL_CONDUCTIVITY,
                               PropertyModel::LINEAR, {0.00006});

    return material;
  }

  // Default material if name not recognized
  return std::make_shared<Material>(MaterialType::FLUID, materialName);
}

/**
 * @brief Set a property value with unit conversion
 * @param property The property to set
 * @param value The property value in specified units
 * @param unitStr The unit string (e.g., "kg/m³", "Pa·s", "W/(m·K)")
 */
void Material::setPropertyWithUnits(MaterialProperty property, double value,
                                    const std::string &unitStr) {
  // Convert value to SI units
  double siValue;

  switch (property) {
  case MaterialProperty::DENSITY:
    siValue = Units::convert(value, unitStr, "kg/m³");
    break;
  case MaterialProperty::DYNAMIC_VISCOSITY:
    siValue = Units::convert(value, unitStr, "Pa·s");
    break;
  case MaterialProperty::THERMAL_CONDUCTIVITY:
    siValue = Units::convert(value, unitStr, "W/(m·K)");
    break;
  case MaterialProperty::SPECIFIC_HEAT:
    siValue = Units::convert(value, unitStr, "J/(kg·K)");
    break;
  default:
    // For properties without specific unit handling, use value directly
    siValue = value;
  }

  // Set the property with the SI value
  setProperty(property, siValue);
}

/**
 * @brief Get property value with unit conversion
 * @param property The property to get
 * @param unitStr The unit string to convert to
 * @return The property value in requested units
 */
double Material::getPropertyWithUnits(MaterialProperty property,
                                      const std::string &unitStr) const {
  double siValue = getProperty(property);

  // Convert from SI units to requested units
  switch (property) {
  case MaterialProperty::DENSITY:
    return Units::convert(siValue, "kg/m³", unitStr);
  case MaterialProperty::DYNAMIC_VISCOSITY:
    return Units::convert(siValue, "Pa·s", unitStr);
  case MaterialProperty::THERMAL_CONDUCTIVITY:
    return Units::convert(siValue, "W/(m·K)", unitStr);
  case MaterialProperty::SPECIFIC_HEAT:
    return Units::convert(siValue, "J/(kg·K)", unitStr);
  default:
    // For properties without specific unit handling, return value directly
    return siValue;
  }
}

/**
 * @brief Get property value at specific temperature with unit conversion
 * @param property The property to get
 * @param temperature Temperature value in specified temperature units
 * @param temperatureUnitStr The temperature unit string (e.g., "K", "C", "F")
 * @param propertyUnitStr The property unit string for the return value
 * @return The property value at the given temperature in requested units
 */
double Material::getPropertyAtTemperatureWithUnits(
    MaterialProperty property, double temperature,
    const std::string &temperatureUnitStr,
    const std::string &propertyUnitStr) const {
  double kelvin = Units::convert(temperature, temperatureUnitStr, "K");
  double siValue = getPropertyAtTemperature(property, kelvin);

  // Convert from SI units to requested units
  switch (property) {
  case MaterialProperty::DENSITY:
    return Units::convert(siValue, "kg/m³", propertyUnitStr);
  case MaterialProperty::DYNAMIC_VISCOSITY:
    return Units::convert(siValue, "Pa·s", propertyUnitStr);
  case MaterialProperty::THERMAL_CONDUCTIVITY:
    return Units::convert(siValue, "W/(m·K)", propertyUnitStr);
  case MaterialProperty::SPECIFIC_HEAT:
    return Units::convert(siValue, "J/(kg·K)", propertyUnitStr);
  default:
    // For properties without specific unit handling, return value directly
    return siValue;
  }
}

/**
 * @brief Get the thermal conductivity of the material at the current reference
 * temperature
 * @return Thermal conductivity value in W/(m·K)
 */
double Material::getThermalConductivity() const {
  return getProperty(MaterialProperty::THERMAL_CONDUCTIVITY);
}

/**
 * @brief Get the thermal conductivity of the material at a specific temperature
 * @param temperature The temperature at which to calculate conductivity (K)
 * @return Thermal conductivity value in W/(m·K)
 */
double Material::getThermalConductivity(double temperature) const {
  return getPropertyAtTemperature(MaterialProperty::THERMAL_CONDUCTIVITY,
                                  temperature);
}
