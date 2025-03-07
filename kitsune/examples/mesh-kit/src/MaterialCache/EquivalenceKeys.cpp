/**
 * @file EquivalenceKeys.cpp
 * @brief Implementation of material equivalence key strategies
 */

#include "EquivalenceKeys.h"

//==============================================================================
// StandardEquivalenceKey Implementation
//==============================================================================

StandardEquivalenceKey::StandardEquivalenceKey(double tolerance)
    : m_tolerance(tolerance) {
  // By default, include all properties in comparison
  for (size_t i = 0; i < static_cast<size_t>(Material::MaterialProperty::COUNT);
       ++i) {
    m_propertyFlags[i] = true;
  }

  // Initialize property weights
  for (size_t i = 0; i < static_cast<size_t>(Material::MaterialProperty::COUNT);
       ++i) {
    m_propertyWeights[i] = 1.0;
  }

  // By default, include type and name in comparison
  m_compareType = true;
  m_compareName = true;
  m_compareRefTemperature = true;
  m_compareMixtureComponents = true;
}

bool StandardEquivalenceKey::areEquivalent(
    const std::shared_ptr<Material> &a,
    const std::shared_ptr<Material> &b) const {
  if (a.get() == b.get()) {
    return true; // Same instance
  }

  // Compare type if enabled
  if (m_compareType && a->getType() != b->getType()) {
    return false;
  }

  // Compare name if enabled
  if (m_compareName && a->getName() != b->getName()) {
    return false;
  }

  // Compare reference temperature if enabled
  if (m_compareRefTemperature &&
      !doubleEquals(a->getReferenceTemperature(), b->getReferenceTemperature(),
                    m_tolerance)) {
    return false;
  }

  // Compare base properties
  for (size_t i = 0; i < static_cast<size_t>(Material::MaterialProperty::COUNT);
       ++i) {
    if (m_propertyFlags[i]) {
      auto prop = static_cast<Material::MaterialProperty>(i);
      if (!doubleEquals(a->getProperty(prop), b->getProperty(prop),
                        m_tolerance)) {
        return false;
      }
    }
  }

  // Compare mixture components if enabled
  if (m_compareMixtureComponents) {
    if (a->isMixture() != b->isMixture()) {
      return false;
    }

    if (a->isMixture()) {
      auto componentsA = a->getMixtureComponents();
      auto componentsB = b->getMixtureComponents();

      if (componentsA.size() != componentsB.size()) {
        return false;
      }

      // Compare components (order matters in the current Material
      // implementation)
      for (size_t i = 0; i < componentsA.size(); ++i) {
        if (!doubleEquals(componentsA[i].second, componentsB[i].second,
                          m_tolerance)) {
          return false;
        }

        // Recursively compare component materials
        if (!areEquivalent(componentsA[i].first, componentsB[i].first)) {
          return false;
        }
      }
    }
  }

  // Materials are equivalent if all enabled comparisons pass
  return true;
}

size_t StandardEquivalenceKey::hash(const std::shared_ptr<Material>& material) const {
  // Create a property-specific tolerance array using the global tolerance
  std::array<double, static_cast<size_t>(Material::MaterialProperty::COUNT)> propertyTolerances;
  for (size_t i = 0; i < static_cast<size_t>(Material::MaterialProperty::COUNT); ++i) {
    propertyTolerances[i] = m_tolerance;
  }
  
  // Delegate to the base implementation
  return hashImpl(material, m_tolerance, m_compareType, m_compareName, 
                 m_compareRefTemperature, m_compareMixtureComponents,
                 m_propertyFlags, propertyTolerances);
}

void StandardEquivalenceKey::setTolerance(double tolerance) {
  m_tolerance = tolerance;

  // Record for statistics
  m_toleranceChanges++;
}

double StandardEquivalenceKey::getTolerance() const { return m_tolerance; }

std::string StandardEquivalenceKey::getName() const {
  return "Standard Equivalence Key";
}

void StandardEquivalenceKey::setPropertyComparison(
    Material::MaterialProperty property, bool include) {
  size_t index = static_cast<size_t>(property);
  if (index < static_cast<size_t>(Material::MaterialProperty::COUNT)) {
    m_propertyFlags[index] = include;
  }
}

bool StandardEquivalenceKey::isPropertyCompared(
    Material::MaterialProperty property) const {
  size_t index = static_cast<size_t>(property);
  if (index < static_cast<size_t>(Material::MaterialProperty::COUNT)) {
    return m_propertyFlags[index];
  }
  return false;
}

std::unordered_map<std::string, double>
StandardEquivalenceKey::getStatistics() const {
  return {{"Tolerance", m_tolerance},
          {"Tolerance Changes", static_cast<double>(m_toleranceChanges)},
          {"Properties Compared", countPropertiesCompared()}};
}

void StandardEquivalenceKey::setCompareType(bool compare) {
  m_compareType = compare;
}

void StandardEquivalenceKey::setCompareName(bool compare) {
  m_compareName = compare;
}

void StandardEquivalenceKey::setCompareReferenceTemperature(bool compare) {
  m_compareRefTemperature = compare;
}

void StandardEquivalenceKey::setCompareMixtureComponents(bool compare) {
  m_compareMixtureComponents = compare;
}

void StandardEquivalenceKey::setPropertyWeight(
    Material::MaterialProperty property, double weight) {
  size_t index = static_cast<size_t>(property);
  if (index < static_cast<size_t>(Material::MaterialProperty::COUNT)) {
    m_propertyWeights[index] = weight;
  }
}

double StandardEquivalenceKey::countPropertiesCompared() const {
  double count = 0;
  for (bool flag : m_propertyFlags) {
    if (flag)
      count += 1.0;
  }
  return count;
}

//==============================================================================
// AdaptiveEquivalenceKey Implementation
//==============================================================================

AdaptiveEquivalenceKey::AdaptiveEquivalenceKey(double baseTolerance,
                                               size_t adaptationInterval)
    : StandardEquivalenceKey(baseTolerance),
      m_adaptationInterval(adaptationInterval), m_baseTolerance(baseTolerance) {
  // Initialize property-specific tolerances
  for (size_t i = 0; i < static_cast<size_t>(Material::MaterialProperty::COUNT);
       ++i) {
    m_propertyTolerances[i] = baseTolerance;
  }
}

bool AdaptiveEquivalenceKey::areEquivalent(
    const std::shared_ptr<Material> &a,
    const std::shared_ptr<Material> &b) const {
  if (a.get() == b.get()) {
    return true; // Same instance
  }

  // Track the comparison for adaptation
  recordComparison(a, b);

  // Compare type if enabled
  if (m_compareType && a->getType() != b->getType()) {
    return false;
  }

  // Compare name if enabled
  if (m_compareName && a->getName() != b->getName()) {
    return false;
  }

  // Compare reference temperature if enabled
  if (m_compareRefTemperature &&
      !doubleEquals(a->getReferenceTemperature(), b->getReferenceTemperature(),
                    m_tolerance)) {
    return false;
  }

  // Compare base properties with property-specific tolerances
  for (size_t i = 0; i < static_cast<size_t>(Material::MaterialProperty::COUNT);
       ++i) {
    if (m_propertyFlags[i]) {
      auto prop = static_cast<Material::MaterialProperty>(i);
      if (!doubleEquals(a->getProperty(prop), b->getProperty(prop),
                        m_propertyTolerances[i])) {
        return false;
      }
    }
  }

  // Compare mixture components if enabled
  if (m_compareMixtureComponents) {
    if (a->isMixture() != b->isMixture()) {
      return false;
    }

    if (a->isMixture()) {
      auto componentsA = a->getMixtureComponents();
      auto componentsB = b->getMixtureComponents();

      if (componentsA.size() != componentsB.size()) {
        return false;
      }

      // Compare components (order matters in the current Material
      // implementation)
      for (size_t i = 0; i < componentsA.size(); ++i) {
        if (!doubleEquals(componentsA[i].second, componentsB[i].second,
                          m_tolerance)) {
          return false;
        }

        // Recursively compare component materials
        if (!areEquivalent(componentsA[i].first, componentsB[i].first)) {
          return false;
        }
      }
    }
  }

  // Materials are equivalent if all enabled comparisons pass
  m_hitCount++; // Record a hit for adaptation
  return true;
}

// Implementation of hash method
size_t AdaptiveEquivalenceKey::hash(const std::shared_ptr<Material> &material) const {
  // Delegate to the base implementation, passing our property-specific tolerances
  return hashImpl(material, m_tolerance, m_compareType, m_compareName, 
                 m_compareRefTemperature, m_compareMixtureComponents,
                 m_propertyFlags, m_propertyTolerances);
}

void AdaptiveEquivalenceKey::setTolerance(double tolerance) {
  m_tolerance = tolerance;
  m_baseTolerance = tolerance;

  // Update all property tolerances
  for (size_t i = 0; i < static_cast<size_t>(Material::MaterialProperty::COUNT);
       ++i) {
    m_propertyTolerances[i] = tolerance;
  }

  // Record for statistics
  m_toleranceChanges++;
}

std::string AdaptiveEquivalenceKey::getName() const {
  return "Adaptive Equivalence Key";
}

void AdaptiveEquivalenceKey::setPropertyTolerance(
    Material::MaterialProperty property, double tolerance) {
  size_t index = static_cast<size_t>(property);
  if (index < static_cast<size_t>(Material::MaterialProperty::COUNT)) {
    m_propertyTolerances[index] = tolerance;
  }
}

double AdaptiveEquivalenceKey::getPropertyTolerance(
    Material::MaterialProperty property) const {
  size_t index = static_cast<size_t>(property);
  if (index < static_cast<size_t>(Material::MaterialProperty::COUNT)) {
    return m_propertyTolerances[index];
  }
  return m_tolerance;
}

std::unordered_map<std::string, double>
AdaptiveEquivalenceKey::getStatistics() const {
  double avgTolerance = 0.0;
  for (size_t i = 0; i < static_cast<size_t>(Material::MaterialProperty::COUNT);
       ++i) {
    avgTolerance += m_propertyTolerances[i];
  }
  avgTolerance /= static_cast<double>(Material::MaterialProperty::COUNT);

  return {{"Base Tolerance", m_baseTolerance},
          {"Average Tolerance", avgTolerance},
          {"Tolerance Changes", static_cast<double>(m_toleranceChanges)},
          {"Adaptation Count", static_cast<double>(m_adaptationCount)},
          {"Comparison Count", static_cast<double>(m_comparisonCount)},
          {"Hit Count", static_cast<double>(m_hitCount)},
          {"Hit Rate",
           m_comparisonCount > 0
               ? (static_cast<double>(m_hitCount) / m_comparisonCount) * 100.0
               : 0.0},
          {"Properties Compared", countPropertiesCompared()}};
}

// Fix for the recordComparison method
void AdaptiveEquivalenceKey::recordComparison(
    const std::shared_ptr<Material> &a,
    const std::shared_ptr<Material> &b) const {
  m_comparisonCount++;

  // Record property differences for adaptation
  for (size_t i = 0; i < static_cast<size_t>(Material::MaterialProperty::COUNT);
       ++i) {
    if (m_propertyFlags[i]) {
      auto prop = static_cast<Material::MaterialProperty>(i);
      double valueA = a->getProperty(prop);
      double valueB = b->getProperty(prop);

      // Record absolute difference
      double diff = std::abs(valueA - valueB);
      
      // Also record relative difference for large values
      if (std::abs(valueA) > 1.0 || std::abs(valueB) > 1.0) {
        double maxVal = std::max(std::abs(valueA), std::abs(valueB));
        double relDiff = diff / maxVal;
        
        // Use relative difference for large values - it's more meaningful
        if (maxVal > 100.0) {  // Threshold for "large" values
          diff = relDiff * m_baseTolerance * 1000;  // Scale for comparison
        }
      }
      
      m_propertyVariations[i].record(diff);
    }
  }

  // Adapt more frequently during early usage (aggressive learning)
  if (m_comparisonCount < 100) {
    // Adapt every comparison for first 10 comparisons
    if (m_comparisonCount <= 10) {
      adaptTolerances();
    }
    // Then every 5 comparisons until 100
    else if (m_comparisonCount % 5 == 0) {
      adaptTolerances();
    }
  }
  // Use normal adaptation interval after 100 comparisons
  else if (m_comparisonCount % m_adaptationInterval == 0) {
    adaptTolerances();
  }
}

// Fix for the adaptTolerances method
void AdaptiveEquivalenceKey::adaptTolerances() const {
  m_adaptationCount++;

  // Adapt tolerance for each property based on observed variations
  for (size_t i = 0; i < static_cast<size_t>(Material::MaterialProperty::COUNT);
       ++i) {
    if (!m_propertyFlags[i]) {
      continue;  // Skip properties that aren't compared
    }
    
    const auto &variation = m_propertyVariations[i];
    if (variation.count < 3) {
      continue; // Need at least some samples, but be more aggressive (3 instead of 10)
    }

    double mean = variation.getMean();
    double stdDev = variation.getStandardDeviation();
    auto prop = static_cast<Material::MaterialProperty>(i);

    // Skip properties that have zero or infinite values
    double propValue = mean;
    if (propValue == 0.0 || std::isinf(propValue) || std::isnan(propValue)) {
      continue;
    }

    // If property is density, be more aggressive with adaptation
    if (prop == Material::MaterialProperty::DENSITY) {
      // For the density property specifically, use a much more aggressive approach
      // This helps with the common case of small density variations
      double newTolerance = std::max(m_baseTolerance, mean * 2.0);
      
      // If distribution has some spread, ensure tolerance captures it
      if (stdDev > 0) {
        newTolerance = std::max(newTolerance, stdDev * 3.0);  // Cover 3-sigma range
      }
      
      // Ensure we have a minimum reasonable tolerance for large density values
      if (propValue > 100.0) {
        // For densities around 1000, ensure we have at least 0.01 absolute tolerance
        newTolerance = std::max(newTolerance, propValue * 0.00001);
      }
      
      // Don't decrease tolerance too much from previous value (stability)
      m_propertyTolerances[i] = std::max(m_propertyTolerances[i] * 0.5, newTolerance);
    } 
    else {
      // For other properties, use a more conservative approach
      if (stdDev > 0) {
        // Simple heuristic: set tolerance to capture "noise" but not signal
        // Use a fraction of the standard deviation
        double newTolerance = std::max(m_baseTolerance, mean * 0.5);

        // If distribution is bimodal (indicating separate clusters),
        // set tolerance below the gap between clusters
        if (variation.getRange() > 10 * stdDev) {
          // Bimodal distribution detected
          newTolerance = std::min(newTolerance, mean * 0.1);
        }

        // Don't change tolerance too drastically
        double factor = std::max(0.5, std::min(2.0, newTolerance / m_propertyTolerances[i]));
        m_propertyTolerances[i] = m_propertyTolerances[i] * factor;
      }
    }
  }

  // Adapt which properties to compare based on their significance
  adaptPropertyComparison();
}

// Enhanced version of the PropertyVariation record method
void AdaptiveEquivalenceKey::PropertyVariation::record(double value) {
  // Ignore NaN and Inf values
  if (std::isnan(value) || std::isinf(value)) {
    return;
  }
  
  minValue = std::min(minValue, value);
  maxValue = std::max(maxValue, value);
  
  // Exponential moving average for more stable statistics
  if (count == 0) {
    // First value
    sum = value;
    sumSquared = value * value;
  } else {
    // Use a decay factor that gives more weight to recent values
    double alpha = std::min(0.3, 1.0 / count);  // Decay factor, max 0.3
    sum = (1.0 - alpha) * sum + alpha * value;
    sumSquared = (1.0 - alpha) * sumSquared + alpha * (value * value);
  }
  
  count++;
}

double AdaptiveEquivalenceKey::PropertyVariation::getRange() const {
  return maxValue - minValue;
}

double AdaptiveEquivalenceKey::PropertyVariation::getMean() const {
  return count > 0 ? sum / count : 0.0;
}

double AdaptiveEquivalenceKey::PropertyVariation::getVariance() const {
  if (count < 2)
    return 0.0;
  double mean = getMean();
  return (sumSquared / count) - (mean * mean);
}

double AdaptiveEquivalenceKey::PropertyVariation::getStandardDeviation() const {
  return std::sqrt(getVariance());
}

void AdaptiveEquivalenceKey::adaptPropertyComparison() const {
  // Count properties with meaningful variations
  std::vector<std::pair<size_t, double>> propertyVariances;

  for (size_t i = 0; i < static_cast<size_t>(Material::MaterialProperty::COUNT);
       ++i) {
    const auto &variation = m_propertyVariations[i];
    if (variation.count >= 10) {
      propertyVariances.emplace_back(i, variation.getVariance());
    }
  }

  // Sort by variance (most significant first)
  std::sort(propertyVariances.begin(), propertyVariances.end(),
            [](const auto &a, const auto &b) { return a.second > b.second; });

  // Keep comparing properties with significant variance
  // Disable comparison for properties with near-zero variance
  for (size_t i = 0; i < propertyVariances.size(); ++i) {
    size_t propIndex = propertyVariances[i].first;
    double variance = propertyVariances[i].second;

    // Enable high-variance properties, disable low-variance ones
    if (variance < 1e-20) {
      m_propertyFlags[propIndex] = false; // Effectively constant property
    }
  }
}

//==============================================================================
// DomainSpecificEquivalenceKey Implementation
//==============================================================================

DomainSpecificEquivalenceKey::DomainSpecificEquivalenceKey(
    DomainType domainType, double tolerance)
    : StandardEquivalenceKey(tolerance), m_domainType(domainType) {
  // Configure based on domain type
  configureDomain(domainType);
}

std::string DomainSpecificEquivalenceKey::getName() const {
  return "Domain-Specific Equivalence Key (" + getDomainName(m_domainType) +
         ")";
}

void DomainSpecificEquivalenceKey::setDomainType(DomainType domainType) {
  if (domainType != m_domainType) {
    m_domainType = domainType;
    configureDomain(domainType);
    m_domainChanges++;
  }
}

DomainSpecificEquivalenceKey::DomainType
DomainSpecificEquivalenceKey::getDomainType() const {
  return m_domainType;
}

std::unordered_map<std::string, double>
DomainSpecificEquivalenceKey::getStatistics() const {
  auto stats = StandardEquivalenceKey::getStatistics();
  stats["Domain Changes"] = static_cast<double>(m_domainChanges);
  return stats;
}

void DomainSpecificEquivalenceKey::configureDomain(DomainType domainType) {
  // Reset all properties to default
  for (size_t i = 0; i < static_cast<size_t>(Material::MaterialProperty::COUNT);
       ++i) {
    m_propertyFlags[i] = true;
    m_propertyWeights[i] = 1.0;
  }

  m_compareType = true;
  m_compareName = false; // Generally ignore names for physics-based simulation
  m_compareRefTemperature = true;
  m_compareMixtureComponents = true;

  // Configure specific settings for each domain type
  switch (domainType) {
  case DomainType::THERMAL:
    // Emphasize thermal properties
    m_propertyWeights[static_cast<size_t>(
        Material::MaterialProperty::THERMAL_CONDUCTIVITY)] = 2.0;
    m_propertyWeights[static_cast<size_t>(
        Material::MaterialProperty::SPECIFIC_HEAT)] = 2.0;
    m_propertyWeights[static_cast<size_t>(
        Material::MaterialProperty::THERMAL_EXPANSION)] = 2.0;
    break;

  case DomainType::FLUID_FLOW:
    // Emphasize flow properties
    m_propertyWeights[static_cast<size_t>(
        Material::MaterialProperty::DENSITY)] = 2.0;
    m_propertyWeights[static_cast<size_t>(
        Material::MaterialProperty::DYNAMIC_VISCOSITY)] = 2.0;

    // De-emphasize thermal properties
    m_propertyWeights[static_cast<size_t>(
        Material::MaterialProperty::THERMAL_CONDUCTIVITY)] = 0.5;
    break;

  case DomainType::MULTIPHASE:
    // Strict comparison for multiphase simulations
    m_tolerance /= 10.0; // Tighter tolerance
    m_compareMixtureComponents = true;

    // Emphasize interface properties
    m_propertyWeights[static_cast<size_t>(
        Material::MaterialProperty::SURFACE_TENSION)] = 2.0;
    break;

  case DomainType::THERMAL_STRESS:
    // Emphasize mechanical and thermal properties
    m_propertyWeights[static_cast<size_t>(
        Material::MaterialProperty::THERMAL_EXPANSION)] = 2.0;
    m_propertyWeights[static_cast<size_t>(
        Material::MaterialProperty::THERMAL_CONDUCTIVITY)] = 1.5;

    // Add mechanical properties (if they were added to MaterialProperty)
    // For now, we use the existing ones
    break;

  case DomainType::HIGH_PERFORMANCE:
    // Loose tolerance for performance
    m_tolerance *= 10.0;

    // Only compare essential properties
    m_propertyFlags[static_cast<size_t>(
        Material::MaterialProperty::ELECTRICAL_CONDUCTIVITY)] = false;
    m_propertyFlags[static_cast<size_t>(
        Material::MaterialProperty::SURFACE_TENSION)] = false;

    // Don't compare mixture components deeply
    m_compareMixtureComponents = false;
    break;

  case DomainType::GENERAL:
  default:
    // Balanced settings (already set above)
    break;
  }
}

std::string DomainSpecificEquivalenceKey::getDomainName(DomainType type) const {
  switch (type) {
  case DomainType::THERMAL:
    return "Thermal";
  case DomainType::FLUID_FLOW:
    return "Fluid Flow";
  case DomainType::MULTIPHASE:
    return "Multiphase";
  case DomainType::THERMAL_STRESS:
    return "Thermal Stress";
  case DomainType::HIGH_PERFORMANCE:
    return "High Performance";
  case DomainType::GENERAL:
    return "General";
  default:
    return "Unknown";
  }
}

