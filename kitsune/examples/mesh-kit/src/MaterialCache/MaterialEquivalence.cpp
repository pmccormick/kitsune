/**
 * @file MaterialEquivalence.cpp
 * @brief Implementation of the base MaterialEquivalence class
 *
 * This file implements the methods defined in MaterialEquivalence.h, providing
 * the core functionality for comparing materials and determining equivalence.
 * The implementation focuses on providing robust comparison and hashing methods
 * that work well for the typical property ranges encountered in computational
 * science simulations.
 */

#include "MaterialEquivalence.h"
#include "CacheAnalyticsPolicy.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>

//------------------------------------------------------------------------------
// MaterialEquivalence Implementation
//------------------------------------------------------------------------------

MaterialEquivalence::MaterialEquivalence(double tolerance)
    : m_baseTolerance(tolerance) {
  // By default, include all properties in comparison
  for (size_t i = 0; i < static_cast<size_t>(Material::MaterialProperty::COUNT);
       ++i) {
    m_propertyFlags[i] = true;
    m_propertyTolerances[i] = tolerance;
  }
}

bool MaterialEquivalence::areEquivalent(const std::shared_ptr<Material> &a,
                                        const std::shared_ptr<Material> &b,
                                        CacheAnalyticsPolicy *analytics) const {
  // Quick check for same instance
  if (a.get() == b.get()) {
    // Record this check in analytics if available
    if (analytics) {
      analytics->onEquivalenceCheck(a, b, true);
    }
    return true;
  }

  // Compare type if enabled
  if (m_compareType && a->getType() != b->getType()) {
    // Record this check in analytics if available
    if (analytics) {
      analytics->onEquivalenceCheck(a, b, false);
    }
    return false;
  }

  // Compare name if enabled
  if (m_compareName && a->getName() != b->getName()) {
    // Record this check in analytics if available
    if (analytics) {
      analytics->onEquivalenceCheck(a, b, false);
    }
    return false;
  }

  // Compare reference temperature if enabled
  if (m_compareRefTemperature) {
    if (!compareValues(a->getReferenceTemperature(),
                       b->getReferenceTemperature(), m_baseTolerance)) {
      // Record this check in analytics if available
      if (analytics) {
        analytics->onEquivalenceCheck(a, b, false);
      }
      return false;
    }
  }

  // Compare properties
  for (size_t i = 0; i < static_cast<size_t>(Material::MaterialProperty::COUNT);
       ++i) {
    if (m_propertyFlags[i]) {
      auto prop = static_cast<Material::MaterialProperty>(i);
      double valueA = a->getProperty(prop);
      double valueB = b->getProperty(prop);

      if (!compareValues(valueA, valueB, m_propertyTolerances[i])) {
        // Record this check in analytics if available
        if (analytics) {
          analytics->onEquivalenceCheck(a, b, false);
        }
        return false;
      }
    }
  }

  // Compare mixture components if enabled
  if (m_compareMixtureComponents) {
    if (a->isMixture() != b->isMixture()) {
      // Record this check in analytics if available
      if (analytics) {
        analytics->onEquivalenceCheck(a, b, false);
      }
      return false;
    }

    if (a->isMixture()) {
      auto componentsA = a->getMixtureComponents();
      auto componentsB = b->getMixtureComponents();

      if (componentsA.size() != componentsB.size()) {
        // Record this check in analytics if available
        if (analytics) {
          analytics->onEquivalenceCheck(a, b, false);
        }
        return false;
      }

      // Compare components (order matters in the current implementation)
      for (size_t i = 0; i < componentsA.size(); ++i) {
        if (!compareValues(componentsA[i].second, componentsB[i].second,
                           m_baseTolerance)) {
          // Record this check in analytics if available
          if (analytics) {
            analytics->onEquivalenceCheck(a, b, false);
          }
          return false;
        }

        // Recursively compare component materials
        if (!areEquivalent(componentsA[i].first, componentsB[i].first,
                           analytics)) {
          return false;
        }
      }
    }
  }

  // All comparisons passed
  // Record this successful check in analytics if available
  if (analytics) {
    analytics->onEquivalenceCheck(a, b, true);
  }
  return true;
}

size_t MaterialEquivalence::hash(const std::shared_ptr<Material> &material,
                                 CacheAnalyticsPolicy *analytics) const {
  // Use a prime number for initial hash
  size_t h = 17;

  // Hash the type if comparison is enabled
  if (m_compareType) {
    h = h * 31 + std::hash<int>{}(static_cast<int>(material->getType()));
  }

  // Hash the name if comparison is enabled
  if (m_compareName) {
    h = h * 31 + std::hash<std::string>{}(material->getName());
  }

  // Hash the reference temperature if comparison is enabled
  if (m_compareRefTemperature) {
    h = h * 31 +
        hashValue(material->getReferenceTemperature(), m_baseTolerance);
  }

  // Hash each compared property
  for (size_t i = 0; i < static_cast<size_t>(Material::MaterialProperty::COUNT);
       ++i) {
    if (m_propertyFlags[i]) {
      auto prop = static_cast<Material::MaterialProperty>(i);
      double value = material->getProperty(prop);
      h = h * 31 + hashValue(value, m_propertyTolerances[i]);
    }
  }

  // Hash mixture components if enabled
  if (m_compareMixtureComponents && material->isMixture()) {
    auto components = material->getMixtureComponents();

    // Hash component count
    h = h * 31 + components.size();

    // Hash each component
    for (size_t i = 0; i < components.size(); ++i) {
      // Hash the component's material (recursively)
      h = h * 31 + hash(components[i].first, analytics);

      // Hash the fraction
      h = h * 31 + hashValue(components[i].second, m_baseTolerance);
    }
  }

  // Record this hash in analytics if available
  if (analytics) {
    analytics->onMaterialHash(material, h);
  }

  return h;
}

void MaterialEquivalence::setTolerance(double tolerance) {
  m_baseTolerance = tolerance;

  // Update all property tolerances that match the old base tolerance
  for (size_t i = 0; i < static_cast<size_t>(Material::MaterialProperty::COUNT);
       ++i) {
    if (m_propertyTolerances[i] == m_baseTolerance) {
      m_propertyTolerances[i] = tolerance;
    }
  }
}

double MaterialEquivalence::getTolerance() const { return m_baseTolerance; }

void MaterialEquivalence::setPropertyComparison(
    Material::MaterialProperty property, bool include) {
  size_t index = static_cast<size_t>(property);
  if (index < static_cast<size_t>(Material::MaterialProperty::COUNT)) {
    m_propertyFlags[index] = include;
  }
}

bool MaterialEquivalence::isPropertyCompared(
    Material::MaterialProperty property) const {
  size_t index = static_cast<size_t>(property);
  if (index < static_cast<size_t>(Material::MaterialProperty::COUNT)) {
    return m_propertyFlags[index];
  }
  return false;
}

std::string MaterialEquivalence::getName() const {
  return "Generic Material Equivalence";
}

void MaterialEquivalence::setPropertyTolerance(
    Material::MaterialProperty property, double tolerance) {
  size_t index = static_cast<size_t>(property);
  if (index < static_cast<size_t>(Material::MaterialProperty::COUNT)) {
    m_propertyTolerances[index] = tolerance;
  }
}

double MaterialEquivalence::getPropertyTolerance(
    Material::MaterialProperty property) const {
  size_t index = static_cast<size_t>(property);
  if (index < static_cast<size_t>(Material::MaterialProperty::COUNT)) {
    return m_propertyTolerances[index];
  }
  return m_baseTolerance;
}

bool MaterialEquivalence::compareValues(double a, double b,
                                        double tolerance) const {
  // Handle exact equality
  if (a == b) {
    return true;
  }

  // Handle special cases
  if (std::isnan(a) || std::isnan(b)) {
    return false; // NaN is never equal to anything, including itself
  }

  if (std::isinf(a) && std::isinf(b)) {
    return (a > 0) == (b > 0); // Same-signed infinities are equal
  }

  // Check if the numbers are really close (needed when comparing near zero)
  double diff = std::fabs(a - b);
  if (diff <= tolerance) {
    return true;
  }

  // Use relative difference for larger values
  a = std::fabs(a);
  b = std::fabs(b);
  double largest = (b > a) ? b : a;

  return diff <= largest * tolerance;
}

size_t MaterialEquivalence::hashValue(double value, double tolerance) const {
  // Handle special cases
  if (std::isnan(value)) {
    return std::hash<const char *>{}("NaN");
  }

  if (std::isinf(value)) {
    return std::hash<const char *>{}(value > 0 ? "Infinity" : "-Infinity");
  }

  // For near-zero values, return a constant
  if (std::fabs(value) <= tolerance) {
    return 0;
  }

  // For normal values, quantize based on tolerance
  double quantize = tolerance;

  // For larger values, use relative tolerance
  if (std::fabs(value) > 1.0) {
    quantize *= std::fabs(value);
  }

  // Compute quantized value
  double quantized = std::round(value / quantize) * quantize;

  // Convert to size_t for hashing
  return std::hash<double>{}(quantized);
}
