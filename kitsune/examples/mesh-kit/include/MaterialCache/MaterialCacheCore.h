/**
 * @file MaterialCacheCore.h
 * @brief Core interfaces and structures for the material caching system
 * @details
 *
 * This file provides the core interfaces for the material caching system,
 * including the IMaterialCache interface and supporting classes.
 */

#pragma once

#include "Material.h"
#include <chrono>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declarations
class IMaterialCache;
class IMaterialEquivalenceKey;
class IEvictionPolicy;
class MaterialCacheStatistics;

/**
 * @class IMaterialEquivalenceKey
 * @brief Interface for determining when two materials are equivalent
 */
class IMaterialEquivalenceKey {
public:
  virtual ~IMaterialEquivalenceKey() = default;

  /**
   * @brief Check if two materials are equivalent
   * @param a First material
   * @param b Second material
   * @return True if the materials are functionally equivalent
   */
  virtual bool areEquivalent(const std::shared_ptr<Material> &a,
                             const std::shared_ptr<Material> &b) const = 0;

  /**
   * @brief Get a hash value for a material
   * @param material The material to hash
   * @return A hash value for the material
   */
  virtual size_t hash(const std::shared_ptr<Material> &material) const = 0;

  /**
   * @brief Set the tolerance for equivalence checks
   * @param tolerance The new tolerance value
   */
  virtual void setTolerance(double tolerance) = 0;

  /**
   * @brief Get the current tolerance
   * @return The tolerance value
   */
  virtual double getTolerance() const = 0;

  /**
   * @brief Get the name of this equivalence key
   * @return The name
   */
  virtual std::string getName() const = 0;

  /**
   * @brief Set whether to compare a specific property
   * @param property The property to set comparison for
   * @param include Whether to include this property in comparison
   */
  virtual void setPropertyComparison(Material::MaterialProperty property,
                                     bool include) = 0;

  /**
   * @brief Check if a property is being compared
   * @param property The property to check
   * @return True if the property is included in comparison
   */
  virtual bool
  isPropertyCompared(Material::MaterialProperty property) const = 0;

  /**
   * @brief Get statistics for this equivalence key
   * @return Map of statistic name to value
   */
  virtual std::unordered_map<std::string, double> getStatistics() const = 0;
};

/**
 * @class IEvictionPolicy
 * @brief Interface for cache eviction policies
 */
class IEvictionPolicy {
public:
  virtual ~IEvictionPolicy() = default;

  /**
   * @brief Notify the policy that an item was accessed
   * @param item The item that was accessed
   */
  virtual void onAccess(void *item) = 0;

  /**
   * @brief Select an item for eviction
   * @return Pointer to the item to evict
   */
  virtual void *selectForEviction() = 0;

  /**
   * @brief Add an item to the policy
   * @param item The item to add
   */
  virtual void addItem(void *item) = 0;

  /**
   * @brief Remove an item from the policy
   * @param item The item to remove
   */
  virtual void removeItem(void *item) = 0;

  /**
   * @brief Clear all items from the policy
   */
  virtual void clear() = 0;

  /**
   * @brief Get the number of items in the policy
   * @return Number of items
   */
  virtual size_t size() const = 0;

  /**
   * @brief Get the name of this policy
   * @return The policy name
   */
  virtual std::string getName() const = 0;

  /**
   * @brief Get statistics for this policy
   * @return Map of statistic name to value
   */
  virtual std::unordered_map<std::string, double> getStatistics() const = 0;
};

/**
 * @class IMaterialCache
 * @brief Interface for material caching
 */
class IMaterialCache {
public:
  virtual ~IMaterialCache() = default;

  /**
   * @brief Get a material from the cache or add it if not found
   * @param material The material to look for
   * @return A cached material that is equivalent to the input
   */
  virtual std::shared_ptr<Material>
  getOrAdd(const std::shared_ptr<Material> &material) = 0;

  /**
   * @brief Find a material in the cache
   * @param material The material to look for
   * @return A cached material that is equivalent to the input, or nullptr if
   * not found
   */
  virtual std::shared_ptr<Material>
  find(const std::shared_ptr<Material> &material) = 0;

  /**
   * @brief Add a material to the cache
   * @param material The material to add
   * @return The added material (may be different if an equivalent was already
   * cached)
   */
  virtual std::shared_ptr<Material>
  add(const std::shared_ptr<Material> &material) = 0;

  /**
   * @brief Get the number of materials in the cache
   * @return Number of materials
   */
  virtual size_t size() const = 0;

  /**
   * @brief Clear the cache
   */
  virtual void clear() = 0;

  /**
   * @brief Preload the cache with a set of materials
   * @param materials The materials to add
   */
  virtual void
  preload(const std::vector<std::shared_ptr<Material>> &materials) = 0;

  /**
   * @brief Get cache statistics
   * @return Reference to cache statistics
   */
  virtual const MaterialCacheStatistics &getStatistics() const = 0;

  /**
   * @brief Reset cache statistics
   */
  virtual void resetStatistics() = 0;

  /**
   * @brief Get the equivalence key used by this cache
   * @return Reference to the equivalence key
   */
  virtual const IMaterialEquivalenceKey &getEquivalenceKey() const = 0;

  /**
   * @brief Get the eviction policy used by this cache
   * @return Reference to the eviction policy
   */
  virtual const IEvictionPolicy &getEvictionPolicy() const = 0;

  /**
   * @brief Get the name of this cache
   * @return The cache name
   */
  virtual std::string getName() const = 0;

  /**
   * @brief Set the maximum cache size
   * @param size The new maximum size (0 = unlimited)
   */
  virtual void setMaxSize(size_t size) = 0;

  /**
   * @brief Get the maximum cache size
   * @return The maximum size (0 = unlimited)
   */
  virtual size_t getMaxSize() const = 0;

  /**
   * @brief Set the tolerance for material equivalence
   * @param tolerance The new tolerance value
   */
  virtual void setTolerance(double tolerance) = 0;
};
