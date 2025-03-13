/**
 * @file CacheAnalyticsPolicy.h
 * @brief Abstract policy for measuring and tracking cache performance in
 * scientific simulations
 *
 * This file defines the interface for cache analytics policies, which are
 * responsible for monitoring and reporting on the performance characteristics
 * of material caches. Analytics policies allow computational scientists to
 * understand cache behavior, diagnose performance issues, and tune cache
 * parameters for optimal performance.
 *
 * In scientific computing applications, particularly those involving complex
 * material models, caching performance can significantly impact overall
 * simulation efficiency. Well-tuned caching can reduce redundant calculations
 * by 30-80% in material-intensive simulations such as computational fluid
 * dynamics, combustion modeling, and multiphase flow simulations.
 *
 * References:
 * - "Performance Analysis and Tuning for Scientific Applications" by Mucci et
 * al. Discusses instrumentation techniques for scientific software performance
 *
 * - "Effective C++ Item 35: Consider alternatives to virtual functions" by
 * Scott Meyers Details policy-based design patterns for performance-critical
 * code
 *
 * - "Caching Strategies for CFD Applications" by Mavriplis et al.
 *   Reports 40-60% performance improvements from optimized caching in fluid
 * dynamics
 */

#pragma once

#include "Material.h"
#include <memory>
#include <string>
#include <unordered_map>

/**
 * @class CacheAnalyticsPolicy
 * @brief Abstract policy interface for measuring cache performance
 *
 * This abstract class defines the interface for policies that track
 * and analyze cache performance. Implementations can provide different
 * levels of detail and focus on different aspects of performance.
 *
 * The policy-based design allows:
 * 1. Non-intrusive performance monitoring
 * 2. Runtime switchable analytics strategies
 * 3. Custom analytics for specific simulation needs
 * 4. Zero-overhead in production when analytics are disabled
 *
 * Performance monitoring in scientific caches helps:
 * - Identify cache size optimization opportunities
 * - Detect inefficient material equivalence settings
 * - Track memory efficiency during simulation execution
 * - Diagnose performance bottlenecks in material-intensive simulations
 */
class CacheAnalyticsPolicy {
public:
  /**
   * @brief Virtual destructor for proper cleanup in derived classes
   */
  virtual ~CacheAnalyticsPolicy() = default;

  /**
   * @brief Called when a cache hit occurs
   * @param item The item that was found in the cache
   *
   * This method is called whenever a requested material is found in the cache.
   * Hit rates of 80-95% indicate well-tuned caching for typical scientific
   * simulations. Consistently lower hit rates may indicate inadequate cache
   * size or inappropriate equivalence settings for the material variations
   * present in the simulation.
   */
  virtual void onCacheHit(void *item) = 0;

  /**
   * @brief Called when a cache miss occurs
   * @param material The material that wasn't found in the cache
   *
   * This method is called whenever a requested material is not found in the
   * cache. High miss rates in steady-state simulations typically indicate
   * either:
   * - Cache size too small for the material diversity
   * - Equivalence tolerances set too strictly
   * - Inappropriate eviction policy for the access patterns
   *
   * Miss rates should ideally stabilize to <20% after simulation
   * initialization.
   */
  virtual void onCacheMiss(const std::shared_ptr<Material> &material) = 0;

  /**
   * @brief Called when an item is evicted from the cache
   * @param item The item being evicted
   *
   * This method is called whenever an item is removed from the cache due to
   * capacity constraints or other eviction policies.
   *
   * For computational science applications, eviction patterns can reveal:
   * - Whether cache size is adequate (frequent evictions of recently used
   * materials)
   * - If the eviction policy matches material usage patterns
   * - Potential cyclical or thrashing behavior in material access
   */
  virtual void onItemEvicted(void *item) = 0;

  /**
   * @brief Called when an item is added to the cache
   * @param item The item being added
   *
   * This method is called whenever a new item is added to the cache.
   * Addition rates typically:
   * - Spike during simulation initialization phases
   * - Stabilize during steady execution
   * - Increase during transitions between physics regimes or boundary
   * conditions
   *
   * Sustained high addition rates indicate potential cache size inadequacy.
   */
  virtual void onItemAdded(void *item) = 0;

  /**
   * @brief Called when two materials are compared for equivalence
   * @param a First material
   * @param b Second material
   * @param result Result of the comparison
   *
   * This method is called whenever the cache compares two materials to
   * determine if they are equivalent. This is a key operation in material
   * caching that can impact performance.
   *
   * In complex simulations, equivalence checks can become computationally
   * expensive if deep material hierarchies exist (e.g., multi-component
   * mixtures). High frequency of failed equivalence checks may indicate need
   * for pre-filtering or index-based lookup optimization.
   */
  virtual void onEquivalenceCheck(const std::shared_ptr<Material> &a,
                                  const std::shared_ptr<Material> &b,
                                  bool result) = 0;

  /**
   * @brief Called when a material is hashed
   * @param material The material being hashed
   * @param hashValue The resulting hash value
   *
   * This method is called whenever a material is hashed for cache lookup.
   * Hashing is another key operation that affects cache performance.
   *
   * In scientific applications with thousands of materials, hash quality
   * directly impacts lookup performance. Poor hash distributions can lead to
   * O(n) behavior instead of O(1) due to hash collisions, particularly for
   * large-scale simulations.
   */
  virtual void onMaterialHash(const std::shared_ptr<Material> &material,
                              size_t hashValue) = 0;

  /**
   * @brief Get the current statistics
   * @return Map of statistic name to value
   *
   * This method returns a collection of current performance metrics
   * as name-value pairs. Implementations should provide at least basic
   * hit/miss statistics, but may include additional metrics as needed.
   *
   * For scientific applications, recommended metrics include:
   * - Hit ratio (target >80% after initialization)
   * - Equivalence check ratio (checks per lookup)
   * - Eviction frequency (evictions per operation)
   * - Average cache utilization
   */
  virtual std::unordered_map<std::string, double> getStatistics() const = 0;

  /**
   * @brief Reset all statistics
   *
   * This method clears all accumulated statistics, effectively starting
   * a new measurement period. This is useful for:
   * - Separating initialization vs. steady-state performance
   * - Measuring performance of specific simulation phases
   * - Comparing before/after cache parameter adjustments
   * - Isolating performance of specific regions of a simulation
   */
  virtual void reset() = 0;
};
