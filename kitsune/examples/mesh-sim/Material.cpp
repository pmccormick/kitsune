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

std::shared_ptr<Material> Material::createMixture(std::shared_ptr<Material> other, 
						  double mixFraction, 
						  const std::string& mixingRule) const {
  // Ensure the mix fraction is in valid range
  mixFraction = std::max(0.0, std::min(1.0, mixFraction));
    
  // Create a new material for the mixture
  auto mixture = std::make_shared<Material>(MaterialType::FLUID, 
					    m_name + "-" + other->getName() + "-Mixture");
    
  // Mark as mixture
  mixture->m_isMixture = true;
    
  // If both inputs are already mixtures, combine their components
  if (m_isMixture && other->m_isMixture) {
    // Add components from first material scaled by (1-mixFraction)
    for (const auto& component : m_mixtureComponents) {
      mixture->m_mixtureComponents.push_back({
	  component.first, 
	  component.second * (1.0 - mixFraction)
	});
    }
        
    // Add components from second material scaled by mixFraction
    for (const auto& component : other->m_mixtureComponents) {
      mixture->m_mixtureComponents.push_back({
	  component.first, 
	  component.second * mixFraction
	});
    }
  } 
  else if (m_isMixture) {
    // First material is mixture, second is pure
    for (const auto& component : m_mixtureComponents) {
      mixture->m_mixtureComponents.push_back({
	  component.first, 
	  component.second * (1.0 - mixFraction)
	});
    }
    mixture->m_mixtureComponents.push_back({other, mixFraction});
  }
  else if (other->m_isMixture) {
    // First material is pure, second is mixture
    mixture->m_mixtureComponents.push_back({
	std::make_shared<Material>(*this), 
	1.0 - mixFraction
      });
        
    for (const auto& component : other->m_mixtureComponents) {
      mixture->m_mixtureComponents.push_back({
	  component.first, 
	  component.second * mixFraction
	});
    }
  }
  else {
    // Both materials are pure
    mixture->m_mixtureComponents.push_back({
	std::make_shared<Material>(*this), 
	1.0 - mixFraction
      });
    mixture->m_mixtureComponents.push_back({other, mixFraction});
  }
    
  // Calculate base properties for the mixture at reference temperature
  for (size_t i = 0; i < static_cast<size_t>(MaterialProperty::COUNT); ++i) {
    MaterialProperty prop = static_cast<MaterialProperty>(i);
    double value1 = getPropertyAtTemperature(prop, m_referenceTemperature);
    double value2 = other->getPropertyAtTemperature(prop, m_referenceTemperature);
        
    double mixedValue = mixProperties(prop, value1, value2, mixFraction, mixingRule);
    mixture->setProperty(prop, mixedValue);
  }
    
  return mixture;
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

std::shared_ptr<Material> Material::createPredefined(const std::string& materialName) {
  std::shared_ptr<Material> material;
    
  if (materialName == "water") {
    material = std::make_shared<Material>(MaterialType::FLUID, "Water");
    material->setReferenceTemperature(293.15); // 20°C
        
    // Set base properties at 20°C
    material->setProperty(MaterialProperty::DENSITY, 998.2); // kg/m³
    material->setProperty(MaterialProperty::DYNAMIC_VISCOSITY, 1.0016e-3); // Pa·s
    material->setProperty(MaterialProperty::THERMAL_CONDUCTIVITY, 0.6); // W/(m·K)
    material->setProperty(MaterialProperty::SPECIFIC_HEAT, 4182.0); // J/(kg·K)
    material->setProperty(MaterialProperty::THERMAL_EXPANSION, 2.07e-4); // 1/K
    material->setProperty(MaterialProperty::SURFACE_TENSION, 0.0728); // N/m
        
    // Set temperature-dependent models
    // Density: linear model with coefficient
    material->setPropertyModel(MaterialProperty::DENSITY, PropertyModel::LINEAR, {-0.0002}); // decreases with temp
        
    // Viscosity: exponential model
    material->setPropertyModel(MaterialProperty::DYNAMIC_VISCOSITY, PropertyModel::EXPONENTIAL, {-0.02}); // decreases with temp
        
    // Thermal conductivity: linear model
    material->setPropertyModel(MaterialProperty::THERMAL_CONDUCTIVITY, PropertyModel::LINEAR, {0.0015}); // increases with temp
        
    // Specific heat: polynomial model (coefficients are for delta T in K)
    material->setPropertyModel(MaterialProperty::SPECIFIC_HEAT, PropertyModel::POLYNOMIAL, {0.0005, -0.0000006});
  }
  else if (materialName == "air") {
    material = std::make_shared<Material>(MaterialType::FLUID, "Air");
    material->setReferenceTemperature(293.15); // 20°C
        
    // Set base properties at 20°C
    material->setProperty(MaterialProperty::DENSITY, 1.204); // kg/m³
    material->setProperty(MaterialProperty::DYNAMIC_VISCOSITY, 1.825e-5); // Pa·s
    material->setProperty(MaterialProperty::THERMAL_CONDUCTIVITY, 0.0257); // W/(m·K)
    material->setProperty(MaterialProperty::SPECIFIC_HEAT, 1005.0); // J/(kg·K)
    material->setProperty(MaterialProperty::THERMAL_EXPANSION, 3.43e-3); // 1/K
        
    // Set temperature-dependent models
    // For ideal gas: density inversely proportional to temperature
    material->setCustomPropertyFunction(MaterialProperty::DENSITY, 
					[material](double T) -> double {
					  double rho0 = material->getProperty(MaterialProperty::DENSITY);
					  double T0 = material->getReferenceTemperature();
					  return rho0 * (T0 / T); // assuming constant pressure
					});
        
    // Viscosity: Sutherland's law (simplified)
    material->setCustomPropertyFunction(MaterialProperty::DYNAMIC_VISCOSITY,
					[material](double T) -> double {
					  double mu0 = material->getProperty(MaterialProperty::DYNAMIC_VISCOSITY);
					  double T0 = material->getReferenceTemperature();
					  return mu0 * pow(T / T0, 0.7); // simplified Sutherland
					});
        
    // Thermal conductivity: power law
    material->setPropertyModel(MaterialProperty::THERMAL_CONDUCTIVITY, PropertyModel::LINEAR, {0.00008});
  }
  else if (materialName == "aluminum") {
    material = std::make_shared<Material>(MaterialType::SOLID, "Aluminum");
    material->setReferenceTemperature(293.15); // 20°C
        
    // Set base properties
    material->setProperty(MaterialProperty::DENSITY, 2700.0); // kg/m³
    material->setProperty(MaterialProperty::THERMAL_CONDUCTIVITY, 237.0); // W/(m·K)
    material->setProperty(MaterialProperty::SPECIFIC_HEAT, 900.0); // J/(kg·K)
    material->setProperty(MaterialProperty::THERMAL_EXPANSION, 2.31e-5); // 1/K
    material->setProperty(MaterialProperty::ELECTRICAL_CONDUCTIVITY, 3.5e7); // S/m
        
    // Metals have slight temperature dependence on thermal conductivity
    material->setPropertyModel(MaterialProperty::THERMAL_CONDUCTIVITY, PropertyModel::LINEAR, {-0.0004});
        
    // Specific heat increases slightly with temperature
    material->setPropertyModel(MaterialProperty::SPECIFIC_HEAT, PropertyModel::LINEAR, {0.0005});
  }
  else {
    // Default material
    material = std::make_shared<Material>(MaterialType::FLUID, materialName);
  }
    
  return material;
}

