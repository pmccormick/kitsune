/**
 * @file MaterialCache.h
 * @brief Main implementation of the material caching system
 * @details
 * 
 * This file provides the primary implementation of the material caching
 * system, designed to reduce memory fragmentation and improve performance
 * in CFD simulations.
 */

#pragma once

#include "MaterialCacheCore.h"
#include "MaterialCacheStatistics.h"
#include "EquivalenceKeys.h"
#include "EvictionPolicies.h"
#include "EnhancedStatistics.h"
#include "MaterialBehaviorTracker.h"
#include <unordered_map>
#include <memory>
#include <mutex>
#include <chrono>
#include <atomic>

/**
 * @class MaterialCache
 * @brief Caches Material objects to avoid redundancy and memory fragmentation
 * @details
 * 
 * The MaterialCache class provides the primary implementation of the material caching
 * system, designed to reduce memory fragmentation and improve performance in CFD simulations.
 * It uses the Strategy Pattern to allow customization of both the material equivalence
 * determination and the cache eviction policy.
 * 
 * Key features:
 * 1. Configurable cache size: Limits memory usage while maximizing hit rate
 * 2. Pluggable equivalence strategy: Determines when materials are functionally equivalent
 * 3. Pluggable eviction policy: Determines which materials to remove when the cache is full
 * 4. Thread safety: Supports concurrent access from multiple threads
 * 5. Comprehensive statistics: Tracks hits, misses, and other performance metrics
 * 6. Contamination detection: Identifies materials that are taking up space but rarely used
 */
class MaterialCache : public IMaterialCache {
public:
  /**
   * @brief Constructor with default strategies
   * @param maxSize Maximum cache size (0 = unlimited)
   * @param tolerance Tolerance for material equivalence
   * @param enableContaminationDetection Whether to enable contamination detection
   */
  MaterialCache(size_t maxSize = 100, double tolerance = 1e-6, bool enableContaminationDetection = true);
    
  /**
   * @brief Constructor with explicit strategies
   * @param equivalenceKey Material equivalence strategy
   * @param evictionPolicy Cache eviction policy
   * @param maxSize Maximum cache size (0 = unlimited)
   * @param enableContaminationDetection Whether to enable contamination detection
   */
  MaterialCache(std::unique_ptr<IMaterialEquivalenceKey> equivalenceKey,
		std::unique_ptr<IEvictionPolicy> evictionPolicy,
		size_t maxSize = 100,
		bool enableContaminationDetection = true);
    
  /**
   * @brief Destructor
   */
  ~MaterialCache();
    
  /**
   * @brief Get a material from the cache or add it if not found
   * @param material The material to look for
   * @return A cached material that is equivalent to the input
   */
  std::shared_ptr<Material> getOrAdd(const std::shared_ptr<Material>& material) override;
    
  /**
   * @brief Find a material in the cache
   * @param material The material to look for
   * @return A cached material that is equivalent to the input, or nullptr if not found
   */
  std::shared_ptr<Material> find(const std::shared_ptr<Material>& material) override;
    
  /**
   * @brief Add a material to the cache
   * @param material The material to add
   * @return The added material (may be different if an equivalent was already cached)
   */
  std::shared_ptr<Material> add(const std::shared_ptr<Material>& material) override;
    
  /**
   * @brief Get the number of materials in the cache
   * @return Number of materials
   */
  size_t size() const override;
    
  /**
   * @brief Clear the cache
   */
  void clear() override;
    
  /**
   * @brief Preload the cache with a set of materials
   * @param materials The materials to add
   */
  void preload(const std::vector<std::shared_ptr<Material>>& materials) override;
    
  /**
   * @brief Get cache statistics
   * @return Reference to cache statistics
   */
  const MaterialCacheStatistics& getStatistics() const override;
    
  /**
   * @brief Reset cache statistics
   */
  void resetStatistics() override;
    
  /**
   * @brief Get the equivalence key used by this cache
   * @return Reference to the equivalence key
   */
  const IMaterialEquivalenceKey& getEquivalenceKey() const override;
    
  /**
   * @brief Get the eviction policy used by this cache
   * @return Reference to the eviction policy
   */
  const IEvictionPolicy& getEvictionPolicy() const override;
    
  /**
   * @brief Get the name of this cache
   * @return The cache name
   */
  std::string getName() const override;
    
  /**
   * @brief Set the maximum cache size
   * @param size The new maximum size (0 = unlimited)
   */
  void setMaxSize(size_t size) override;
    
  /**
   * @brief Get the maximum cache size
   * @return The maximum size (0 = unlimited)
   */
  size_t getMaxSize() const override;
    
  /**
   * @brief Set the tolerance for material equivalence
   * @param tolerance The new tolerance value
   */
  void setTolerance(double tolerance) override;
    
  /**
   * @brief Set a new equivalence key strategy
   * @param equivalenceKey The new strategy
   */
  void setEquivalenceKey(std::unique_ptr<IMaterialEquivalenceKey> equivalenceKey);
    
  /**
   * @brief Set a new eviction policy strategy
   * @param evictionPolicy The new strategy
   */
  void setEvictionPolicy(std::unique_ptr<IEvictionPolicy> evictionPolicy);
    
  /**
   * @brief Set whether to enable contamination detection
   * @param enable Whether to enable contamination detection
   */
  void setEnableContaminationDetection(bool enable);
    
  /**
   * @brief Check if contamination detection is enabled
   * @return Whether contamination detection is enabled
   */
  bool isContaminationDetectionEnabled() const;
    
  /**
   * @brief Get enhanced statistics
   * @return Reference to enhanced statistics
   */
  const EnhancedMaterialCacheStatistics& getEnhancedStatistics() const;
    
  /**
   * @brief Get behavior tracker
   * @return Reference to behavior tracker
   */
  const MaterialCacheBehaviorTracker& getBehaviorTracker() const;
    
  /**
   * @brief Get contaminator materials
   * @return Vector of material IDs identified as contaminators
   */
  std::vector<uint32_t> getContaminatorMaterials() const;
    
  /**
   * @brief Get detailed information about materials with specific behavior
   * @param behavior The behavior type to query
   * @return Vector of material behavior details
   */
  std::vector<MaterialCacheBehaviorTracker::MaterialBehaviorInfo> 
  getMaterialsWithBehavior(MaterialCacheBehaviorTracker::BehaviorType behavior) const;
    
  /**
   * @brief Get all material behavior statistics
   * @return Vector of all material behavior details
   */
  std::vector<MaterialCacheBehaviorTracker::MaterialBehaviorInfo> getAllMaterialBehaviors() const;
    
  /**
   * @brief Detect and report potential cache contamination
   * @return Vector of materials identified as contaminators
   */
  std::vector<MaterialCacheBehaviorTracker::MaterialBehaviorInfo> detectContamination() const;
    
  /**
   * @brief Evict materials identified as contaminators
   * @param maxToEvict Maximum number of materials to evict (0 = all)
   * @return Number of materials evicted
   */
  size_t evictContaminators(size_t maxToEvict = 0);
    
  /**
   * @brief Get combined detailed statistics
   * @return Map of all statistics including enhanced and behavior tracking
   */
  std::unordered_map<std::string, double> getDetailedStatistics() const;
    
protected:

  // For access from derived classes
  mutable std::mutex m_mutex;

  // Strategies
  std::unique_ptr<IMaterialEquivalenceKey> m_equivalenceKey;
  std::unique_ptr<IEvictionPolicy> m_evictionPolicy;

private:

  struct MaterialPtrHash {
    IMaterialEquivalenceKey* eqKey;
    
    MaterialPtrHash(IMaterialEquivalenceKey* key) : eqKey(key) {}
    
    size_t operator()(const std::shared_ptr<Material>& material) const {
      // Use the equivalence key's hash function directly
      // This ensures consistency between hashing and equality checks
      return eqKey->hash(material);
    }
  };

  struct MaterialPtrEqual {
    IMaterialEquivalenceKey* eqKey;
    
    MaterialPtrEqual(IMaterialEquivalenceKey* key) : eqKey(key) {}
    
    bool operator()(const std::shared_ptr<Material>& a, const std::shared_ptr<Material>& b) const {
      // Use the equivalence key's equality function directly
      return eqKey->areEquivalent(a, b);
    }
  };
  
  // Cache storage
  std::unordered_map<std::shared_ptr<Material>, std::shared_ptr<Material>,
		     MaterialPtrHash, MaterialPtrEqual> m_cache;
  // Configuration
  size_t m_maxSize;
  bool m_enableContaminationDetection;
    
  // Statistics
  MaterialCacheStatistics m_stats;
  EnhancedMaterialCacheStatistics m_enhancedStats;
  MaterialCacheBehaviorTracker m_behaviorTracker;
    
  // Evict an item based on the current policy
  void evictItem();
    
  // Rebuild the cache with new equivalence key
  void rebuildCache();
    
  // Rebuild the eviction policy data
  void rebuildEvictionData();
    
  // Update memory usage statistics
  void updateMemoryUsage();
};
