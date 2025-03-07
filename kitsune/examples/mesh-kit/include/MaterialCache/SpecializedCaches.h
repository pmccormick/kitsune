/**
 * @file SpecializedCaches.h
 * @brief Specialized material cache implementations
 * @details
 *
 * This file provides specialized implementations of MaterialCache for
 * specific use cases, such as fixed-size and adaptive caches.
 */

#pragma once

#include "EquivalenceKeys.h"
#include "EvictionPolicies.h"
#include "MaterialCache.h"
#include <atomic>
#include <vector>

/**
 * @class FixedSizeMaterialCache
 * @brief Material cache with a fixed size
 * @details
 *
 * This cache has a fixed size and cannot be resized. It's optimized for
 * performance in environments with limited memory.
 */
class FixedSizeMaterialCache : public MaterialCache {
public:
  /**
   * @brief Constructor with default strategies
   * @param size Fixed cache size
   * @param tolerance Tolerance for material equivalence
   */
  FixedSizeMaterialCache(size_t size, double tolerance = 1e-6);

  /**
   * @brief Constructor with explicit strategies
   * @param size Fixed cache size
   * @param equivalenceKey Material equivalence strategy
   * @param evictionPolicy Cache eviction policy
   */
  FixedSizeMaterialCache(
      size_t size, std::unique_ptr<IMaterialEquivalenceKey> equivalenceKey,
      std::unique_ptr<IEvictionPolicy> evictionPolicy);

  /**
   * @brief Set the maximum cache size (no-op for fixed size cache)
   * @param size The new maximum size (ignored)
   */
  void setMaxSize(size_t size) override;

  /**
   * @brief Get the name of this cache
   * @return The cache name
   */
  std::string getName() const override;
};

/**
 * @class AdaptiveMaterialCache
 * @brief Material cache with adaptive sizing and strategies
 * @details
 *
 * This cache automatically adjusts its size and strategies based on observed
 * usage patterns, optimizing for both memory usage and performance.
 */
class AdaptiveMaterialCache : public MaterialCache {
public:
  /**
   * @brief Constructor
   * @param initialSize Initial cache size
   * @param maxSize Maximum cache size (0 = unlimited)
   * @param initialTolerance Initial tolerance for material equivalence
   * @param adaptationInterval Number of operations between adaptations
   */
  AdaptiveMaterialCache(size_t initialSize = 100, size_t maxSize = 1000,
                        double initialTolerance = 1e-6,
                        size_t adaptationInterval = 1000);

  /**
   * @brief Get a material from the cache or add it if not found
   * @param material The material to look for
   * @return A cached material that is equivalent to the input
   */
  std::shared_ptr<Material>
  getOrAdd(const std::shared_ptr<Material> &material) override;

  /**
   * @brief Get the name of this cache
   * @return The cache name
   */
  std::string getName() const override;

  /**
   * @brief Get additional statistics
   * @return Map of statistic name to value
   */
  std::unordered_map<std::string, double> getAdaptiveStatistics() const;

private:
  size_t m_initialSize;
  size_t m_maxSize;
  size_t m_adaptationInterval;

  // Adaptation state
  std::atomic<size_t> m_operationsSinceLastAdaptation{0};
  size_t m_adaptationCount = 0;
  size_t m_sizeAdaptationCount = 0;
  size_t m_configAdaptationCount = 0;

  // History for hit rates
  std::vector<double> m_hitRateHistory;
  const size_t MAX_HISTORY = 10;

  // Last adaptation times
  size_t m_lastSizeAdaptation;
  size_t m_lastConfigAdaptation;

  // Adapt the cache based on observed patterns
  void adaptCache();

  // Adapt the cache size
  void adaptCacheSize();

  // Adapt the cache configuration
  void adaptCacheConfiguration();
};
