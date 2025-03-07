/**
 * @file EquivalenceKeys.h
 * @brief Material equivalence key implementations
 * @details
 *
 * This file provides implementations of the IMaterialEquivalenceKey interface
 * for determining when two Material objects should be considered functionally
 * equivalent. It includes:
 *
 * - StandardEquivalenceKey: Basic implementation with configurable tolerance
 * - AdaptiveEquivalenceKey: Self-tuning implementation that adjusts tolerance
 * - DomainSpecificEquivalenceKey: Pre-configured for specific CFD domains
 *
 * It also includes utility functions for floating-point comparison and hashing.
 */

#pragma once

#include "Material.h"
#include "MaterialCacheCore.h"
#include "BaseEquivalanceKey.h"
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
/**
 * @brief Get a hash value for a double considering tolerance
 * @param value The double value to hash
 * @param tolerance The comparison tolerance
 * @return A discretized hash value
 */
inline size_t hashDouble(double value, double tolerance) {
  if (std::isnan(value)) {
    return std::hash<const char*>{}("NaN");
  }

  if (std::isinf(value)) {
    return std::hash<const char*>{}(value > 0 ? "Infinity" : "-Infinity");
  }

  // Discretize the value based on tolerance
  if (tolerance > 0.0) {
    // For large values, use relative tolerance
    if (std::abs(value) > 1.0) {
      // Use a relative tolerance approach for large values
      double bucketSize = tolerance * std::abs(value);
      value = std::round(value / bucketSize) * bucketSize;
    } else {
      // Use absolute tolerance for smaller values
      value = std::round(value / tolerance) * tolerance;
    }
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

  // For values close to zero, use absolute tolerance
  if (std::abs(a) < 1.0 || std::abs(b) < 1.0) {
    return std::abs(a - b) <= tolerance;
  }
  
  // For larger values, use a combined approach with relative tolerance
  double absDiff = std::abs(a - b);
  double absMax = std::max(std::abs(a), std::abs(b));
  
  // Check both absolute and relative difference
  return (absDiff <= tolerance) || (absDiff / absMax <= tolerance);
}

/**
 * @class StandardEquivalenceKey
 * @brief Standard implementation of IMaterialEquivalenceKey
 * @details
 *
 * The StandardEquivalenceKey class provides a configurable implementation of
 * the IMaterialEquivalenceKey interface, determining when two Material objects
 * should be considered functionally equivalent within a specified tolerance.
 *
 * Key features:
 * 1. Configurable tolerance: Controls the precision of floating-point
 * comparisons
 * 2. Selective property comparison: Allows enabling/disabling comparison of
 * specific properties
 * 3. Property weighting: Assigns importance weights to different properties
 * 4. Customizable comparison aspects: Controls whether to compare type, name,
 * etc.
 */
class StandardEquivalenceKey : public BaseEquivalenceKey {
public:
  /**
   * @brief Constructor
   * @param tolerance The floating point comparison tolerance
   */
  explicit StandardEquivalenceKey(double tolerance = 1e-6);

  /**
   * @brief Check if two materials are equivalent
   * @param a First material
   * @param b Second material
   * @return True if the materials are functionally equivalent
   */
  bool areEquivalent(const std::shared_ptr<Material> &a,
                     const std::shared_ptr<Material> &b) const override;

  /**
   * @brief Get a hash value for a material
   * @param material The material to hash
   * @return A hash value for the material
   */
  size_t hash(const std::shared_ptr<Material> &material) const override;

  /**
   * @brief Get the name of this equivalence key
   * @return The name
   */
  std::string getName() const override;

  /**
   * @brief Set whether to compare a specific property
   * @param property The property to set comparison for
   * @param include Whether to include this property in comparison
   */
  void setPropertyComparison(Material::MaterialProperty property,
                             bool include) override;

  /**
   * @brief Check if a property is being compared
   * @param property The property to check
   * @return True if the property is included in comparison
   */
  bool isPropertyCompared(Material::MaterialProperty property) const override;

  /**
   * @brief Get statistics for this equivalence key
   * @return Map of statistic name to value
   */
  std::unordered_map<std::string, double> getStatistics() const override;

  /**
   * @brief Enable or disable comparing material types
   * @param compare Whether to compare types
   */
  void setCompareType(bool compare);

  /**
   * @brief Enable or disable comparing material names
   * @param compare Whether to compare names
   */
  void setCompareName(bool compare);

  /**
   * @brief Enable or disable comparing reference temperatures
   * @param compare Whether to compare reference temperatures
   */
  void setCompareReferenceTemperature(bool compare);

  /**
   * @brief Enable or disable comparing mixture components
   * @param compare Whether to compare mixture components
   */
  void setCompareMixtureComponents(bool compare);

  /**
   * @brief Set the weight for a property in comparison
   * @param property The property to set weight for
   * @param weight The weight (0.0 to 1.0)
   */
  void setPropertyWeight(Material::MaterialProperty property, double weight);

protected:
  mutable std::array<bool, static_cast<size_t>(Material::MaterialProperty::COUNT)>
      m_propertyFlags;
  std::array<double, static_cast<size_t>(Material::MaterialProperty::COUNT)>
      m_propertyWeights;
  bool m_compareType;
  bool m_compareName;
  bool m_compareRefTemperature;
  bool m_compareMixtureComponents;

  // Statistics
  mutable size_t m_toleranceChanges = 0;

  double countPropertiesCompared() const;
};

/**
 * @class AdaptiveEquivalenceKey
 * @brief Self-tuning implementation of material equivalence determination
 * @details
 *
 * The AdaptiveEquivalenceKey class extends StandardEquivalenceKey to provide a
 * self-tuning implementation that automatically adjusts tolerance and
 * comparison settings based on observed material variations during simulation.
 *
 * Key features:
 * 1. Automatic tolerance adjustment: Adapts to the observed distribution of
 * property values
 * 2. Property-specific tolerances: Uses different tolerances for different
 * properties
 * 3. Statistical analysis: Identifies meaningful variations vs. numerical noise
 * 4. Adaptive property selection: Automatically enables/disables properties
 * based on significance
 */
class AdaptiveEquivalenceKey : public StandardEquivalenceKey {
public:
  /**
   * @brief Constructor
   * @param baseTolerance Initial tolerance value
   * @param adaptationInterval Number of comparisons between adaptations
   */
  AdaptiveEquivalenceKey(double baseTolerance = 1e-6,
                         size_t adaptationInterval = 1000);

  /**
   * @brief Check if two materials are equivalent
   * @param a First material
   * @param b Second material
   * @return True if the materials are functionally equivalent
   */
  bool areEquivalent(const std::shared_ptr<Material> &a,
                     const std::shared_ptr<Material> &b) const override;

  /**
   * @brief Get a hash value for a material
   * @param material The material to hash
   * @return A hash value for the material
   */
  size_t hash(const std::shared_ptr<Material> &material) const override;

  /**
   * @brief Set the tolerance for equivalence checks
   * @param tolerance The new tolerance value
   */
  void setTolerance(double tolerance) override;

  /**
   * @brief Get the name of this equivalence key
   * @return The name
   */
  std::string getName() const override;

  /**
   * @brief Set tolerance for a specific property
   * @param property The property to set tolerance for
   * @param tolerance The tolerance value
   */
  void setPropertyTolerance(Material::MaterialProperty property,
                            double tolerance);

  /**
   * @brief Get tolerance for a specific property
   * @param property The property to get tolerance for
   * @return The tolerance value
   */
  double getPropertyTolerance(Material::MaterialProperty property) const;

  /**
   * @brief Get statistics for this equivalence key
   * @return Map of statistic name to value
   */
  std::unordered_map<std::string, double> getStatistics() const override;

private:
  size_t m_adaptationInterval;
  double m_baseTolerance;
  mutable std::array<double, static_cast<size_t>(Material::MaterialProperty::COUNT)>
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

    void record(double value);
    double getRange() const;
    double getMean() const;
    double getVariance() const;
    double getStandardDeviation() const;
  };

  mutable std::array<PropertyVariation,
                     static_cast<size_t>(Material::MaterialProperty::COUNT)>
      m_propertyVariations;

  void recordComparison(const std::shared_ptr<Material> &a,
                        const std::shared_ptr<Material> &b) const;

  void adaptTolerances() const;

  void adaptPropertyComparison() const;
};

/**
 * @class DomainSpecificEquivalenceKey
 * @brief Specialized equivalence key for CFD simulation domains
 * @details
 *
 * The DomainSpecificEquivalenceKey class extends StandardEquivalenceKey to
 * provide pre-configured settings optimized for different types of CFD
 * simulations. It allows users to quickly select appropriate comparison
 * settings for their specific domain without manual configuration.
 *
 * Key features:
 * 1. Domain presets: Pre-configured settings for common CFD simulation types
 * 2. Physics-based weighting: Property importance weights based on physical
 * significance
 * 3. Domain-appropriate tolerances: Tolerance settings matched to domain
 * requirements
 * 4. Easy selection: Simple interface to switch between domains
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
                               double tolerance = 1e-6);

  /**
   * @brief Get the name of this equivalence key
   * @return The name
   */
  std::string getName() const override;

  /**
   * @brief Change the domain type
   * @param domainType The new domain type
   */
  void setDomainType(DomainType domainType);

  /**
   * @brief Get the current domain type
   * @return The domain type
   */
  DomainType getDomainType() const;

  /**
   * @brief Get statistics for this equivalence key
   * @return Map of statistic name to value
   */
  std::unordered_map<std::string, double> getStatistics() const override;

private:
  DomainType m_domainType;
  size_t m_domainChanges = 0;

  void configureDomain(DomainType domainType);
  std::string getDomainName(DomainType type) const;
};

