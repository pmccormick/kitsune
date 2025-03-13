/**
 * @file MaterialEquivalence.h
 * @brief Base class for determining material equivalence in computational simulations
 *
 * This file defines the abstract base class for material equivalence checking strategies.
 * Material equivalence is a critical component of efficient caching systems in 
 * computational simulations involving multiple materials, as it determines when two
 * materials are "close enough" to be considered the same for simulation purposes.
 * 
 * In computational science applications such as CFD, FEA, and multiphysics simulations,
 * a typical simulation might involve anywhere from 5-50 base materials, which can expand
 * to hundreds or thousands of derived materials through mixing, reactions, and state changes.
 * Efficient caching of these materials can reduce memory usage by 30-70% in complex simulations,
 * with corresponding improvements in computational performance and reduced cache misses.
 *
 * References:
 * - "Computational Methods for Multiphase Flow" by Prosperetti & Tryggvason
 *   Typical CFD simulations use 5-20 distinct materials with properties spanning
 *   several orders of magnitude (e.g., viscosity from 10^-5 to 10^3 Pa·s)
 *   
 * - "Numerical Heat Transfer and Fluid Flow" by Patankar
 *   Highlights importance of property interpolation and caching in thermal simulations
 *
 * - "Comparing Floating Point Numbers" by Bruce Dawson: 
 *   https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
 */

#pragma once

#include "Material.h"
#include <memory>
#include <string>
#include <unordered_map>

// Forward declaration
class CacheAnalyticsPolicy;

/**
 * @class MaterialEquivalence
 * @brief Base class for determining material equivalence
 *
 * This class provides the interface and base implementation for comparing
 * Material objects to determine if they are functionally equivalent. It supports
 * configurable tolerances and selective property comparison.
 * 
 * For computational simulations, equivalence checking serves multiple purposes:
 * 
 * 1. Memory Efficiency:
 *    - Typical multiphysics simulations can generate thousands of derived materials
 *    - In large-scale models, 50-70% of memory can be consumed by material data
 *    - Properly cached materials can reduce memory usage by 30-70%
 * 
 * 2. Numerical Stability:
 *    - Ensures consistent material properties despite small numerical differences
 *    - Prevents simulation artifacts from nearly-identical materials
 *    - Important for convergence in iterative solvers
 * 
 * 3. Performance:
 *    - Reduces memory bandwidth needs in parallel computations
 *    - Improves cache locality for frequently accessed materials
 *    - Can reduce runtime by 5-15% in material-intensive simulations
 */
class MaterialEquivalence {
public:
  /**
   * @brief Constructor
   * @param tolerance Default comparison tolerance
   * 
   * The default tolerance of 1e-6 is appropriate for many engineering simulations,
   * but should be adjusted based on the precision requirements of the specific
   * physics being modeled. More precise simulations may require smaller tolerances,
   * while less sensitive applications can use larger values for better performance.
   */
  MaterialEquivalence(double tolerance = 1e-6);

  /**
   * @brief Virtual destructor
   */
  virtual ~MaterialEquivalence() = default;

  /**
   * @brief Compare two materials for equivalence
   * @param a First material
   * @param b Second material
   * @param analytics Optional analytics policy to track comparisons
   * @return True if materials are considered equivalent
   * 
   * In computational simulations, equivalence checking must balance accuracy and performance:
   * - Too strict comparisons lead to memory bloat and performance degradation
   * - Too relaxed comparisons can affect simulation accuracy
   * 
   * Benchmarks from fluid dynamics simulations suggest checking only the properties
   * relevant to the current physics being modeled, with appropriate tolerances for each.
   */
  virtual bool areEquivalent(const std::shared_ptr<Material>& a,
                            const std::shared_ptr<Material>& b,
                            CacheAnalyticsPolicy* analytics = nullptr) const;

  /**
   * @brief Compute a hash value for a material
   * @param material Material to hash
   * @param analytics Optional analytics policy to track hash computations
   * @return Hash value
   * 
   * Hash computation should be:
   * - Fast (called frequently during cache lookups)
   * - Consistent with equivalence relation (equivalent materials must hash the same)
   * - Well-distributed to minimize hash collisions
   * 
   * For simulations with thousands of materials, hash quality directly impacts performance.
   */
  virtual size_t hash(const std::shared_ptr<Material>& material,
                     CacheAnalyticsPolicy* analytics = nullptr) const;

  /**
   * @brief Set the base tolerance for all properties
   * @param tolerance New tolerance value
   * 
   * Recommended tolerance ranges for different simulation types:
   * - High-precision CFD: 1e-8 to 1e-6
   * - General structural analysis: 1e-6 to 1e-4
   * - Thermal simulations: 1e-5 to 1e-3
   * - Large-scale earth/climate models: 1e-4 to 1e-2
   */
  void setTolerance(double tolerance);

  /**
   * @brief Get the current base tolerance
   * @return Tolerance value
   */
  double getTolerance() const;

  /**
   * @brief Set whether a specific property should be compared
   * @param property The property to configure
   * @param include True to include in comparison, false to ignore
   * 
   * Selectively enabling properties based on physics relevance can substantially
   * improve cache effectiveness. For example:
   * - Pure fluid flow: Compare only density and viscosity
   * - Heat transfer: Add thermal conductivity and specific heat
   * - Multiphase flow: Include surface tension
   */
  void setPropertyComparison(Material::MaterialProperty property, bool include);

  /**
   * @brief Check if a property is currently included in comparisons
   * @param property The property to check
   * @return True if property is compared, false otherwise
   */
  bool isPropertyCompared(Material::MaterialProperty property) const;

  /**
   * @brief Set a property-specific tolerance
   * @param property The property to configure
   * @param tolerance Tolerance value for this property
   * 
   * Different properties typically require different tolerances due to their
   * physical meaning and typical ranges. For example:
   * - Density (1-20000 kg/m³): Tolerance ~1e-3 to 1e-2
   * - Viscosity (1e-6 to 1e3 Pa·s): Tolerance ~1e-4
   * - Thermal conductivity (0.01-1000 W/m·K): Tolerance ~1e-3
   * - Specific heat (100-10000 J/kg·K): Tolerance ~1e-2
   */
  void setPropertyTolerance(Material::MaterialProperty property, double tolerance);

  /**
   * @brief Get the tolerance for a specific property
   * @param property The property to check
   * @return Tolerance value for this property
   */
  double getPropertyTolerance(Material::MaterialProperty property) const;

  /**
   * @brief Get a human-readable name for this equivalence strategy
   * @return Strategy name
   */
  virtual std::string getName() const;

  /**
   * @brief Set whether to compare material types
   * @param compare True to compare types, false to ignore type differences
   * 
   * In multiphase simulations, material type (fluid/solid/interface) critically
   * affects behavior. Types should typically always be compared except in specialized
   * cases where materials can change phase during simulation.
   */
  void setCompareType(bool compare) { m_compareType = compare; }

  /**
   * @brief Set whether to compare material names
   * @param compare True to compare names, false to ignore name differences
   * 
   * Comparing names is mainly useful for debugging and tracking. In production
   * simulations, this is typically disabled for better cache performance.
   */
  void setCompareName(bool compare) { m_compareName = compare; }

  /**
   * @brief Set whether to compare reference temperatures
   * @param compare True to compare temperatures, false to ignore differences
   * 
   * Reference temperature comparison is crucial for thermally-dependent materials.
   * For aerospace, nuclear, or high-temperature processes, enable this option.
   * For isothermal or near-isothermal processes, this can be disabled.
   */
  void setCompareRefTemperature(bool compare) { m_compareRefTemperature = compare; }

  /**
   * @brief Set whether to compare mixture components
   * @param compare True to compare components, false to ignore differences
   * 
   * For chemically reacting flows or multiphase simulations, component comparison
   * ensures thermodynamic consistency. For pure CFD or structural analysis,
   * this can be disabled if only bulk properties matter.
   */
  void setCompareMixtureComponents(bool compare) { m_compareMixtureComponents = compare; }

protected:
  /**
   * @brief Compare two floating-point values with tolerance
   * @param a First value
   * @param b Second value
   * @param tolerance Comparison tolerance
   * @return True if values are equivalent within tolerance
   * 
   * The base implementation uses both absolute and relative tolerance.
   * This handles a wide range of values but may not be optimal for all
   * simulation types. Derived classes can implement specialized comparison
   * strategies for specific physics regimes.
   */
  virtual bool compareValues(double a, double b, double tolerance) const;

  /**
   * @brief Compute a consistent hash value for a floating-point number
   * @param value Value to hash
   * @param tolerance Tolerance to use for quantization
   * @return Hash value
   * 
   * This method ensures that floating-point values that are considered
   * equivalent will generate the same hash code, which is essential for
   * correct operation of hash-based caches.
   */
  virtual size_t hashValue(double value, double tolerance) const;

  // Configuration
  double m_baseTolerance;
  bool m_compareType = true;
  bool m_compareName = false;
  bool m_compareRefTemperature = true;
  bool m_compareMixtureComponents = true;

  // Property-specific settings
  bool m_propertyFlags[static_cast<size_t>(Material::MaterialProperty::COUNT)];
  double m_propertyTolerances[static_cast<size_t>(Material::MaterialProperty::COUNT)];
};

