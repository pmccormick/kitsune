/**
 * @file MaterialCacheFactory.h
 * @brief Factory methods for creating material cache instances
 * @details
 *
 * This file provides the MaterialCacheFactory class which contains factory
 * methods for creating common cache configurations for different use cases.
 */

#pragma once

#include "EquivalenceKeys.h"
#include "EvictionPolicies.h"
#include "MaterialCache.h"
#include "MaterialCacheCore.h"
#include "SpecializedCaches.h"
#include <memory>

/**
 * @class MaterialCacheFactory
 * @brief Factory for creating different types of material caches
 */
class MaterialCacheFactory {
public:
  /**
   * @brief Create a standard material cache
   * @param maxSize Maximum cache size (0 = unlimited)
   * @param tolerance Tolerance for material equivalence
   * @param enableContaminationDetection Whether to enable contamination
   * detection
   * @return New material cache instance
   */
  static std::unique_ptr<IMaterialCache>
  createStandardCache(size_t maxSize = 100, double tolerance = 1e-6,
                      bool enableContaminationDetection = true);

  /**
   * @brief Create a fixed size material cache
   * @param size Fixed cache size
   * @param tolerance Tolerance for material equivalence
   * @param enableContaminationDetection Whether to enable contamination
   * detection
   * @return New material cache instance
   */
  static std::unique_ptr<IMaterialCache>
  createFixedSizeCache(size_t size, double tolerance = 1e-6,
                       bool enableContaminationDetection = true);

  /**
   * @brief Create an adaptive material cache
   * @param initialSize Initial cache size
   * @param maxSize Maximum cache size (0 = unlimited)
   * @param initialTolerance Initial tolerance for material equivalence
   * @param enableContaminationDetection Whether to enable contamination
   * detection
   * @return New material cache instance
   */
  static std::unique_ptr<IMaterialCache>
  createAdaptiveCache(size_t initialSize = 100, size_t maxSize = 1000,
                      double initialTolerance = 1e-6,
                      bool enableContaminationDetection = true);

  /**
   * @brief Create a domain-specific cache for CFD simulations
   * @param domainType The type of CFD simulation
   * @param maxSize Maximum cache size
   * @param tolerance Tolerance for material equivalence
   * @param enableContaminationDetection Whether to enable contamination
   * detection
   * @return New material cache instance
   */
  static std::unique_ptr<IMaterialCache>
  createDomainSpecificCache(DomainSpecificEquivalenceKey::DomainType domainType,
                            size_t maxSize = 100, double tolerance = 1e-6,
                            bool enableContaminationDetection = true);

  /**
   * @brief Create a high-performance cache for performance-critical simulations
   * @param maxSize Maximum cache size
   * @param enableContaminationDetection Whether to enable contamination
   * detection
   * @return New material cache instance optimized for performance
   */
  static std::unique_ptr<IMaterialCache>
  createHighPerformanceCache(size_t maxSize = 100,
                             bool enableContaminationDetection = true);

  /**
   * @brief Create a memory-optimized cache for large simulations
   * @param maxSize Maximum cache size
   * @param enableContaminationDetection Whether to enable contamination
   * detection
   * @return New material cache instance optimized for memory efficiency
   */
  static std::unique_ptr<IMaterialCache>
  createMemoryOptimizedCache(size_t maxSize = 100,
                             bool enableContaminationDetection = true);

  /**
   * @brief Create a custom cache with specific strategies
   * @param equivalenceKey Material equivalence strategy
   * @param evictionPolicy Cache eviction policy
   * @param maxSize Maximum cache size
   * @param enableContaminationDetection Whether to enable contamination
   * detection
   * @return New material cache instance with custom strategies
   */
  static std::unique_ptr<IMaterialCache>
  createCustomCache(std::unique_ptr<IMaterialEquivalenceKey> equivalenceKey,
                    std::unique_ptr<IEvictionPolicy> evictionPolicy,
                    size_t maxSize = 100,
                    bool enableContaminationDetection = true);
};

