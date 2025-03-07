/**
 * @file MaterialCacheStatistics.h
 * @brief Statistics tracking for material cache performance
 * @details
 *
 * This file provides the MaterialCacheStatistics class which tracks
 * basic performance metrics for the material cache.
 */

#pragma once

#include <chrono>
#include <cstddef>
#include <string>
#include <unordered_map>

/**
 * @class MaterialCacheStatistics
 * @brief Statistics for material cache performance
 */
class MaterialCacheStatistics {
public:
  /**
   * @brief Constructor
   */
  MaterialCacheStatistics() = default;

  /**
   * @brief Record a cache hit
   */
  void recordHit();

  /**
   * @brief Record a cache miss
   */
  void recordMiss();

  /**
   * @brief Record a cache eviction
   */
  void recordEviction();

  /**
   * @brief Record query time
   * @param duration Query duration
   */
  void recordQueryTime(std::chrono::nanoseconds duration);

  /**
   * @brief Update memory usage
   * @param bytes Current memory usage in bytes
   */
  void updateMemoryUsage(size_t bytes);

  /**
   * @brief Get hit count
   * @return Number of cache hits
   */
  size_t getHits() const;

  /**
   * @brief Get miss count
   * @return Number of cache misses
   */
  size_t getMisses() const;

  /**
   * @brief Get eviction count
   * @return Number of cache evictions
   */
  size_t getEvictions() const;

  /**
   * @brief Get total query count
   * @return Total number of queries (hits + misses)
   */
  size_t getTotalQueries() const;

  /**
   * @brief Get hit rate
   * @return Hit rate as percentage (0-100)
   */
  double getHitRate() const;

  /**
   * @brief Get average query time
   * @return Average query time in nanoseconds
   */
  double getAvgQueryTimeNs() const;

  /**
   * @brief Get memory usage
   * @return Memory usage in bytes
   */
  size_t getMemoryUsageBytes() const;

  /**
   * @brief Get all statistics
   * @return Map of statistic name to value
   */
  virtual std::unordered_map<std::string, double> getStatistics() const;

  /**
   * @brief Reset all statistics
   */
  virtual void reset();

protected:
  size_t m_hits = 0;
  size_t m_misses = 0;
  size_t m_evictions = 0;
  size_t m_totalQueries = 0;
  double m_totalQueryTimeNs = 0;
  double m_avgQueryTimeNs = 0;
  size_t m_memoryUsageBytes = 0;
};
