/**
 * @file EnhancedStatistics.h
 * @brief Enhanced statistics for material cache with material-specific insights
 * @details
 *
 * This file provides the EnhancedMaterialCacheStatistics class which extends
 * the base MaterialCacheStatistics with material-specific tracking and
 * analysis.
 */

#pragma once

#include "Material.h"
#include "MaterialCacheStatistics.h"
#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

/**
 * @class EnhancedMaterialCacheStatistics
 * @brief Extended statistics for material cache with material-specific insights
 * @details
 *
 * This class extends the base MaterialCacheStatistics to provide
 * material-specific insights, including tracking per-material type
 * statistics, identifying top hit/miss materials, and other
 * detailed metrics useful for diagnosing cache behavior.
 */
class EnhancedMaterialCacheStatistics : public MaterialCacheStatistics {
public:
  /**
   * @brief Constructor
   */
  EnhancedMaterialCacheStatistics();

  /**
   * @brief Record a cache hit with material information
   * @param material The material that was accessed
   */
  void recordHit(const std::shared_ptr<Material> &material);

  /**
   * @brief Record a cache miss with material information
   * @param material The material that was not found
   */
  void recordMiss(const std::shared_ptr<Material> &material);

  /**
   * @brief Record an eviction with material information
   * @param material The material that was evicted
   */
  void recordEviction(const std::shared_ptr<Material> &material);

  /**
   * @brief Get hit rate for a specific material type
   * @param type The material type
   * @return Hit rate percentage (0-100)
   */
  double getHitRateForType(Material::MaterialType type) const;

  /**
   * @brief Get hit rate for a specific material name
   * @param name The material name
   * @return Hit rate percentage (0-100)
   */
  double getHitRateForName(const std::string &name) const;

  /**
   * @brief Get detailed statistics including material-specific metrics
   * @return Map of statistic name to value
   */
  std::unordered_map<std::string, double> getDetailedStatistics() const;

  /**
   * @brief Get the top materials by hit count
   * @return Vector of material info sorted by hit count (highest first)
   */
  std::vector<std::pair<std::string, size_t>> getTopHitMaterials() const;

  /**
   * @brief Get the top materials by miss count
   * @return Vector of material info sorted by miss count (highest first)
   */
  std::vector<std::pair<std::string, size_t>> getTopMissMaterials() const;

  /**
   * @brief Reset all statistics
   */
  void reset() override;

private:
  /**
   * @struct MaterialTypeStats
   * @brief Statistics specific to a material type
   */
  struct MaterialTypeStats {
    size_t hits = 0;
    size_t misses = 0;
    size_t evictions = 0;
  };

  /**
   * @struct MaterialCounter
   * @brief Tracks access counts for individual materials
   */
  struct MaterialCounter {
    std::string name;
    uint32_t id;
    size_t count;
  };

  std::unordered_map<Material::MaterialType, MaterialTypeStats> m_typeStats;
  std::unordered_map<std::string, MaterialTypeStats> m_materialNameStats;
  std::vector<MaterialCounter> m_topHitMaterials;
  std::vector<MaterialCounter> m_topMissMaterials;

  /**
   * @brief Update the top materials tracking list
   * @param topList The list to update
   * @param material The material to add or update
   * @param maxItems The maximum number of items to track
   */
  void updateTopMaterials(std::vector<MaterialCounter> &topList,
                          const std::shared_ptr<Material> &material,
                          size_t maxItems);
};
