#pragma once

#include "MaterialCacheCore.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

/**
 * @brief Get a hash value for a double considering tolerance
 * @param value The double value to hash
 * @param tolerance The comparison tolerance
 * @return A discretized hash value
 */
inline size_t hashDouble(double value, double tolerance) {
  if (std::isnan(value)) {
    return std::hash<const char *>{}("NaN");
  }

  if (std::isinf(value)) {
    return std::hash<const char *>{}(value > 0 ? "Infinity" : "-Infinity");
  }

  // Discretize the value based on tolerance
  // This ensures that values within tolerance hash to the same bucket
  if (tolerance > 0.0) {
    value = std::round(value / tolerance) * tolerance;
  }

  // Use the standard hash for double
  return std::hash<double>{}(value);
}

/**
 * @brief Check if two doubles are equal within tolerance
 * @param a First value
 * @param b Second value
 * @param tolerance The comparison tolerance
 * @return True if the values are equal within tolerance
 */
inline bool doubleEquals(double a, double b, double tolerance) {
  if (std::isnan(a) && std::isnan(b)) {
    return true; // NaN equals NaN for our purposes
  }

  if (std::isinf(a) && std::isinf(b)) {
    return (a > 0 && b > 0) ||
           (a < 0 && b < 0); // Same-signed infinities are equal
  }

  return std::abs(a - b) <= tolerance;
}

/**
 * @class StandardEquivalenceKey
 * @brief Standard implementation of IMaterialEquivalenceKey
 *
 * Compares materials based on their properties with a configurable tolerance.
 */
class StandardEquivalenceKey : public IMaterialEquivalenceKey {
public:
  /**
   * @brief Constructor
   * @param tolerance The floating point comparison tolerance
   */
  explicit StandardEquivalenceKey(double tolerance = 1e-6)
      : m_tolerance(tolerance) {
    // By default, include all properties in comparison
    for (size_t i = 0;
         i < static_cast<size_t>(Material::MaterialProperty::COUNT); ++i) {
      m_propertyFlags[i] = true;
    }

    // Initialize property weights
    for (size_t i = 0;
         i < static_cast<size_t>(Material::MaterialProperty::COUNT); ++i) {
      m_propertyWeights[i] = 1.0;
    }

    // By default, include type and name in comparison
    m_compareType = true;
    m_compareName = true;
    m_compareRefTemperature = true;
    m_compareMixtureComponents = true;
  }

  bool areEquivalent(const std::shared_ptr<Material> &a,
                     const std::shared_ptr<Material> &b) const override {
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
        !doubleEquals(a->getReferenceTemperature(),
                      b->getReferenceTemperature(), m_tolerance)) {
      return false;
    }

    // Compare base properties
    for (size_t i = 0;
         i < static_cast<size_t>(Material::MaterialProperty::COUNT); ++i) {
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

  size_t hash(const std::shared_ptr<Material> &material) const override {
    size_t h = 0;

    // Include type in hash if enabled
    if (m_compareType) {
      h ^= std::hash<int>{}(static_cast<int>(material->getType()));
    }

    // Include name in hash if enabled
    if (m_compareName) {
      h ^= std::hash<std::string>{}(material->getName());
    }

    // Include reference temperature in hash if enabled
    if (m_compareRefTemperature) {
      h ^= hashDouble(material->getReferenceTemperature(), m_tolerance);
    }

    // Include base properties in hash
    for (size_t i = 0;
         i < static_cast<size_t>(Material::MaterialProperty::COUNT); ++i) {
      if (m_propertyFlags[i]) {
        auto prop = static_cast<Material::MaterialProperty>(i);
        h ^= hashDouble(material->getProperty(prop), m_tolerance) +
             i; // Add i to distinguish properties
      }
    }

    // Include mixture components in hash if enabled
    if (m_compareMixtureComponents && material->isMixture()) {
      auto components = material->getMixtureComponents();

      // Hash component count
      h ^= std::hash<size_t>{}(components.size());

      // Hash each component
      for (size_t i = 0; i < components.size(); ++i) {
        h ^= hash(components[i].first); // Recursive hash of component material
        h ^= hashDouble(components[i].second, m_tolerance); // Hash fraction
      }
    }

    return h;
  }

  void setTolerance(double tolerance) override {
    m_tolerance = tolerance;

    // Record for statistics
    m_toleranceChanges++;
  }

  double getTolerance() const override { return m_tolerance; }

  std::string getName() const override { return "Standard Equivalence Key"; }

  void setPropertyComparison(Material::MaterialProperty property,
                             bool include) override {
    size_t index = static_cast<size_t>(property);
    if (index < static_cast<size_t>(Material::MaterialProperty::COUNT)) {
      m_propertyFlags[index] = include;
    }
  }

  bool isPropertyCompared(Material::MaterialProperty property) const override {
    size_t index = static_cast<size_t>(property);
    if (index < static_cast<size_t>(Material::MaterialProperty::COUNT)) {
      return m_propertyFlags[index];
    }
    return false;
  }

  /**
   * @brief Enable or disable comparing material types
   * @param compare Whether to compare types
   */
  void setCompareType(bool compare) { m_compareType = compare; }

  /**
   * @brief Enable or disable comparing material names
   * @param compare Whether to compare names
   */
  void setCompareName(bool compare) { m_compareName = compare; }

  /**
   * @brief Enable or disable comparing reference temperatures
   * @param compare Whether to compare reference temperatures
   */
  void setCompareReferenceTemperature(bool compare) {
    m_compareRefTemperature = compare;
  }

  /**
   * @brief Enable or disable comparing mixture components
   * @param compare Whether to compare mixture components
   */
  void setCompareMixtureComponents(bool compare) {
    m_compareMixtureComponents = compare;
  }

  /**
   * @brief Set the weight for a property in comparison
   * @param property The property to set weight for
   * @param weight The weight (0.0 to 1.0)
   */
  void setPropertyWeight(Material::MaterialProperty property, double weight) {
    size_t index = static_cast<size_t>(property);
    if (index < static_cast<size_t>(Material::MaterialProperty::COUNT)) {
      m_propertyWeights[index] = weight;
    }
  }

  std::unordered_map<std::string, double> getStatistics() const override {
    return {{"Tolerance", m_tolerance},
            {"Tolerance Changes", static_cast<double>(m_toleranceChanges)},
            {"Properties Compared", countPropertiesCompared()}};
  }

protected:
  double m_tolerance;
  std::array<bool, static_cast<size_t>(Material::MaterialProperty::COUNT)>
      m_propertyFlags;
  std::array<double, static_cast<size_t>(Material::MaterialProperty::COUNT)>
      m_propertyWeights;
  bool m_compareType;
  bool m_compareName;
  bool m_compareRefTemperature;
  bool m_compareMixtureComponents;

  // Statistics
  mutable size_t m_toleranceChanges = 0;

  double countPropertiesCompared() const {
    double count = 0;
    for (bool flag : m_propertyFlags) {
      if (flag)
        count += 1.0;
    }
    return count;
  }
};

/**
 * @class AdaptiveEquivalenceKey
 * @brief Adaptive implementation of IMaterialEquivalenceKey
 *
 * Automatically adjusts tolerance and comparison properties based on observed
 * material variations and usage patterns.
 */
class AdaptiveEquivalenceKey : public StandardEquivalenceKey {
public:
  /**
   * @brief Constructor
   * @param baseTolerance Initial tolerance value
   * @param adaptationInterval Number of comparisons between adaptations
   */
  AdaptiveEquivalenceKey(double baseTolerance = 1e-6,
                         size_t adaptationInterval = 1000)
      : StandardEquivalenceKey(baseTolerance),
        m_adaptationInterval(adaptationInterval),
        m_baseTolerance(baseTolerance) {
    // Initialize property-specific tolerances
    for (size_t i = 0;
         i < static_cast<size_t>(Material::MaterialProperty::COUNT); ++i) {
      m_propertyTolerances[i] = baseTolerance;
    }
  }

  bool areEquivalent(const std::shared_ptr<Material> &a,
                     const std::shared_ptr<Material> &b) const override {
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
        !doubleEquals(a->getReferenceTemperature(),
                      b->getReferenceTemperature(), m_tolerance)) {
      return false;
    }

    // Compare base properties with property-specific tolerances
    for (size_t i = 0;
         i < static_cast<size_t>(Material::MaterialProperty::COUNT); ++i) {
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

  size_t hash(const std::shared_ptr<Material> &material) const override {
    size_t h = 0;

    // Include type in hash if enabled
    if (m_compareType) {
      h ^= std::hash<int>{}(static_cast<int>(material->getType()));
    }

    // Include name in hash if enabled
    if (m_compareName) {
      h ^= std::hash<std::string>{}(material->getName());
    }

    // Include reference temperature in hash if enabled
    if (m_compareRefTemperature) {
      h ^= hashDouble(material->getReferenceTemperature(), m_tolerance);
    }

    // Include base properties in hash with property-specific tolerances
    for (size_t i = 0;
         i < static_cast<size_t>(Material::MaterialProperty::COUNT); ++i) {
      if (m_propertyFlags[i]) {
        auto prop = static_cast<Material::MaterialProperty>(i);
        h ^= hashDouble(material->getProperty(prop), m_propertyTolerances[i]) +
             i;
      }
    }

    // Include mixture components in hash if enabled
    if (m_compareMixtureComponents && material->isMixture()) {
      auto components = material->getMixtureComponents();

      // Hash component count
      h ^= std::hash<size_t>{}(components.size());

      // Hash each component
      for (size_t i = 0; i < components.size(); ++i) {
        h ^= hash(components[i].first); // Recursive hash of component material
        h ^= hashDouble(components[i].second, m_tolerance); // Hash fraction
      }
    }

    return h;
  }

  void setTolerance(double tolerance) override {
    m_tolerance = tolerance;
    m_baseTolerance = tolerance;

    // Update all property tolerances
    for (size_t i = 0;
         i < static_cast<size_t>(Material::MaterialProperty::COUNT); ++i) {
      m_propertyTolerances[i] = tolerance;
    }

    // Record for statistics
    m_toleranceChanges++;
  }

  std::string getName() const override { return "Adaptive Equivalence Key"; }

  /**
   * @brief Set tolerance for a specific property
   * @param property The property to set tolerance for
   * @param tolerance The tolerance value
   */
  void setPropertyTolerance(Material::MaterialProperty property,
                            double tolerance) {
    size_t index = static_cast<size_t>(property);
    if (index < static_cast<size_t>(Material::MaterialProperty::COUNT)) {
      m_propertyTolerances[index] = tolerance;
    }
  }

  /**
   * @brief Get tolerance for a specific property
   * @param property The property to get tolerance for
   * @return The tolerance value
   */
  double getPropertyTolerance(Material::MaterialProperty property) const {
    size_t index = static_cast<size_t>(property);
    if (index < static_cast<size_t>(Material::MaterialProperty::COUNT)) {
      return m_propertyTolerances[index];
    }
    return m_tolerance;
  }

  std::unordered_map<std::string, double> getStatistics() const override {
    double avgTolerance = 0.0;
    for (size_t i = 0;
         i < static_cast<size_t>(Material::MaterialProperty::COUNT); ++i) {
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

private:
  size_t m_adaptationInterval;
  double m_baseTolerance;
  std::array<double, static_cast<size_t>(Material::MaterialProperty::COUNT)>
      m_propertyTolerances;

  // Statistics for adaptation
  mutable size_t m_comparisonCount = 0;
  mutable size_t m_hitCount = 0;
  mutable size_t m_adaptationCount = 0;

  // Property variation tracking
  struct PropertyVariation {
    double minValue = std::numeric_limits<double>::max();
    double maxValue = std::numeric_limits<double>::lowest();
    double sum = 0.0;
    double sumSquared = 0.0;
    size_t count = 0;

    void record(double value) {
      minValue = std::min(minValue, value);
      maxValue = std::max(maxValue, value);
      sum += value;
      sumSquared += value * value;
      count++;
    }

    double getRange() const { return maxValue - minValue; }

    double getMean() const { return count > 0 ? sum / count : 0.0; }

    double getVariance() const {
      if (count < 2)
        return 0.0;
      double mean = getMean();
      return (sumSquared / count) - (mean * mean);
    }

    double getStandardDeviation() const { return std::sqrt(getVariance()); }
  };

  mutable std::array<PropertyVariation,
                     static_cast<size_t>(Material::MaterialProperty::COUNT)>
      m_propertyVariations;

  void recordComparison(const std::shared_ptr<Material> &a,
                        const std::shared_ptr<Material> &b) const {
    m_comparisonCount++;

    // Record property differences for adaptation
    for (size_t i = 0;
         i < static_cast<size_t>(Material::MaterialProperty::COUNT); ++i) {
      auto prop = static_cast<Material::MaterialProperty>(i);
      double valueA = a->getProperty(prop);
      double valueB = b->getProperty(prop);

      // Record absolute difference
      double diff = std::abs(valueA - valueB);
      m_propertyVariations[i].record(diff);
    }

    // Adapt if necessary
    if (m_comparisonCount % m_adaptationInterval == 0) {
      adaptTolerances();
    }
  }

  void adaptTolerances() const {
    m_adaptationCount++;

    // Adapt tolerance for each property based on observed variations
    for (size_t i = 0;
         i < static_cast<size_t>(Material::MaterialProperty::COUNT); ++i) {
      const auto &variation = m_propertyVariations[i];
      if (variation.count < 10)
        continue; // Need enough samples

      double mean = variation.getMean();
      double stdDev = variation.getStandardDeviation();

      // Identify clusters in the differences
      // Tolerance should be above noise level but below meaningful differences
      if (stdDev > 0) {
        // Simple heuristic: set tolerance to capture "noise" but not signal
        // Use a fraction of the standard deviation
        double newTolerance = std::max(m_baseTolerance, mean * 0.1);

        // If distribution is bimodal (indicating separate clusters),
        // set tolerance below the gap between clusters
        if (variation.getRange() > 100 * stdDev) {
          // Bimodal distribution detected
          newTolerance = std::min(newTolerance, mean * 0.01);
        }

        // Don't change tolerance too drastically
        double factor = std::max(
            0.5, std::min(2.0, newTolerance / m_propertyTolerances[i]));
        m_propertyTolerances[i] = m_propertyTolerances[i] * factor;
      }
    }

    // Adapt which properties to compare based on their significance
    adaptPropertyComparison();
  }

  void adaptPropertyComparison() const {
    // Count properties with meaningful variations
    std::vector<std::pair<size_t, double>> propertyVariances;

    for (size_t i = 0;
         i < static_cast<size_t>(Material::MaterialProperty::COUNT); ++i) {
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
};

/**
 * @class DomainSpecificEquivalenceKey
 * @brief Specialized equivalence key for CFD simulation domains
 *
 * Provides domain-specific equivalence rules for common CFD problem types.
 */
class DomainSpecificEquivalenceKey : public StandardEquivalenceKey {
public:
  /**
   * @enum DomainType
   * @brief Different simulation domain types
   */
  enum class DomainType {
    GENERAL,         ///< General purpose - balanced settings
    THERMAL,         ///< Thermal simulation - emphasize thermal properties
    FLUID_FLOW,      ///< Fluid flow simulation - emphasize flow properties
    MULTIPHASE,      ///< Multiphase simulation - strict mixture comparison
    THERMAL_STRESS,  ///< Thermal stress - emphasize mechanical properties
    HIGH_PERFORMANCE ///< Performance-focused - loose tolerance but fast
  };

  /**
   * @brief Constructor
   * @param domainType The simulation domain type
   * @param tolerance Base tolerance value
   */
  DomainSpecificEquivalenceKey(DomainType domainType = DomainType::GENERAL,
                               double tolerance = 1e-6)
      : StandardEquivalenceKey(tolerance), m_domainType(domainType) {
    // Configure based on domain type
    configureDomain(domainType);
  }

  std::string getName() const override {
    return "Domain-Specific Equivalence Key (" + getDomainName(m_domainType) +
           ")";
  }

  /**
   * @brief Change the domain type
   * @param domainType The new domain type
   */
  void setDomainType(DomainType domainType) {
    if (domainType != m_domainType) {
      m_domainType = domainType;
      configureDomain(domainType);
      m_domainChanges++;
    }
  }

  /**
   * @brief Get the current domain type
   * @return The domain type
   */
  DomainType getDomainType() const { return m_domainType; }

  std::unordered_map<std::string, double> getStatistics() const override {
    auto stats = StandardEquivalenceKey::getStatistics();
    stats["Domain Changes"] = static_cast<double>(m_domainChanges);
    return stats;
  }

private:
  DomainType m_domainType;
  size_t m_domainChanges = 0;

  void configureDomain(DomainType domainType) {
    // Reset all properties to default
    for (size_t i = 0;
         i < static_cast<size_t>(Material::MaterialProperty::COUNT); ++i) {
      m_propertyFlags[i] = true;
      m_propertyWeights[i] = 1.0;
    }

    m_compareType = true;
    m_compareName =
        false; // Generally ignore names for physics-based simulation
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

  std::string getDomainName(DomainType type) const {
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
};

