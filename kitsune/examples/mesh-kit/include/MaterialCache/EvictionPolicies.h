/**
 * @file EvictionPolicies.h
 * @brief Cache eviction policy implementations
 * @details
 *
 * This file provides implementations of the IEvictionPolicy interface for
 * determining which materials to remove from the cache when it reaches
 * capacity. It includes several policy implementations optimized for different
 * access patterns:
 *
 * - LRUEvictionPolicy: Least Recently Used (temporal locality)
 * - MRUEvictionPolicy: Most Recently Used (cyclical patterns)
 * - FIFOEvictionPolicy: First In First Out (simple, deterministic)
 * - FrequencyEvictionPolicy: Least Frequently Used (popularity-based)
 * - TimeSensitiveEvictionPolicy: Combines recency and frequency
 * - AdaptiveEvictionPolicy: Auto-selects best policy based on patterns
 */

#pragma once

#include "MaterialCacheCore.h"
#include <algorithm>
#include <chrono>
#include <list>
#include <mutex>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

/**
 * @class LRUEvictionPolicy
 * @brief Least Recently Used eviction policy
 * @details
 *
 * The LRUEvictionPolicy class implements the classic Least Recently Used (LRU)
 * cache eviction strategy, which removes the least recently accessed items when
 * the cache reaches capacity. This is one of the most widely used cache
 * replacement algorithms due to its good performance across a wide range of
 * workloads.
 *
 * Key characteristics:
 * 1. Temporal locality: Assumes recently used items are likely to be used again
 * 2. Age-based: Eviction priority based solely on time since last access
 * 3. O(1) time complexity: Constant time for all operations with proper
 * implementation
 * 4. Simple and effective: Good general-purpose performance
 */
class LRUEvictionPolicy : public IEvictionPolicy {
public:
  /**
   * @brief Notify the policy that an item was accessed
   * @param item The item that was accessed
   */
  void onAccess(void *item) override;

  /**
   * @brief Select an item for eviction
   * @return Pointer to the item to evict
   */
  void *selectForEviction() override;

  /**
   * @brief Add an item to the policy
   * @param item The item to add
   */
  void addItem(void *item) override;

  /**
   * @brief Remove an item from the policy
   * @param item The item to remove
   */
  void removeItem(void *item) override;

  /**
   * @brief Clear all items from the policy
   */
  void clear() override;

  /**
   * @brief Get the number of items in the policy
   * @return Number of items
   */
  size_t size() const override;

  /**
   * @brief Get the name of this policy
   * @return The policy name
   */
  std::string getName() const override;

  /**
   * @brief Get statistics for this policy
   * @return Map of statistic name to value
   */
  std::unordered_map<std::string, double> getStatistics() const override;

private:
  std::list<void *> m_lruList;
  std::unordered_map<void *, std::list<void *>::iterator> m_itemMap;
  size_t m_accessCount = 0;
};

/**
 * @class MRUEvictionPolicy
 * @brief Most Recently Used eviction policy
 * @details
 *
 * Evicts the most recently accessed items first.
 * This can be useful for scenarios with cyclic access patterns.
 */
class MRUEvictionPolicy : public IEvictionPolicy {
public:
  /**
   * @brief Notify the policy that an item was accessed
   * @param item The item that was accessed
   */
  void onAccess(void *item) override;

  /**
   * @brief Select an item for eviction
   * @return Pointer to the item to evict
   */
  void *selectForEviction() override;

  /**
   * @brief Add an item to the policy
   * @param item The item to add
   */
  void addItem(void *item) override;

  /**
   * @brief Remove an item from the policy
   * @param item The item to remove
   */
  void removeItem(void *item) override;

  /**
   * @brief Clear all items from the policy
   */
  void clear() override;

  /**
   * @brief Get the number of items in the policy
   * @return Number of items
   */
  size_t size() const override;

  /**
   * @brief Get the name of this policy
   * @return The policy name
   */
  std::string getName() const override;

  /**
   * @brief Get statistics for this policy
   * @return Map of statistic name to value
   */
  std::unordered_map<std::string, double> getStatistics() const override;

private:
  std::list<void *> m_mruList;
  std::unordered_map<void *, std::list<void *>::iterator> m_itemMap;
  size_t m_accessCount = 0;
};

/**
 * @class FIFOEvictionPolicy
 * @brief First In First Out eviction policy
 * @details
 *
 * Evicts items in the order they were added, regardless of access patterns.
 */
class FIFOEvictionPolicy : public IEvictionPolicy {
public:
  /**
   * @brief Notify the policy that an item was accessed
   * @param item The item that was accessed
   */
  void onAccess(void *item) override;

  /**
   * @brief Select an item for eviction
   * @return Pointer to the item to evict
   */
  void *selectForEviction() override;

  /**
   * @brief Add an item to the policy
   * @param item The item to add
   */
  void addItem(void *item) override;

  /**
   * @brief Remove an item from the policy
   * @param item The item to remove
   */
  void removeItem(void *item) override;

  /**
   * @brief Clear all items from the policy
   */
  void clear() override;

  /**
   * @brief Get the number of items in the policy
   * @return Number of items
   */
  size_t size() const override;

  /**
   * @brief Get the name of this policy
   * @return The policy name
   */
  std::string getName() const override;

  /**
   * @brief Get statistics for this policy
   * @return Map of statistic name to value
   */
  std::unordered_map<std::string, double> getStatistics() const override;

private:
  std::list<void *> m_fifoQueue;
  std::unordered_set<void *> m_itemSet;
  size_t m_accessCount = 0;
};

/**
 * @class FrequencyEvictionPolicy
 * @brief Frequency-based eviction policy
 * @details
 *
 * Evicts the least frequently accessed items first (LFU).
 * This is useful for workloads with stable popularity patterns.
 */
class FrequencyEvictionPolicy : public IEvictionPolicy {
public:
  /**
   * @brief Notify the policy that an item was accessed
   * @param item The item that was accessed
   */
  void onAccess(void *item) override;

  /**
   * @brief Select an item for eviction
   * @return Pointer to the item to evict
   */
  void *selectForEviction() override;

  /**
   * @brief Add an item to the policy
   * @param item The item to add
   */
  void addItem(void *item) override;

  /**
   * @brief Remove an item from the policy
   * @param item The item to remove
   */
  void removeItem(void *item) override;

  /**
   * @brief Clear all items from the policy
   */
  void clear() override;

  /**
   * @brief Get the number of items in the policy
   * @return Number of items
   */
  size_t size() const override;

  /**
   * @brief Get the name of this policy
   * @return The policy name
   */
  std::string getName() const override;

  /**
   * @brief Get statistics for this policy
   * @return Map of statistic name to value
   */
  std::unordered_map<std::string, double> getStatistics() const override;

private:
  std::unordered_map<void *, size_t> m_frequencyMap;
  std::vector<std::pair<void *, size_t>> m_frequencyList;
  size_t m_accessCount = 0;

  /**
   * @brief Update the frequency list for an item
   * @param item The item to update
   */
  void updateFrequencyList(void *item);

  /**
   * @brief Sort the frequency list
   */
  void sortFrequencyList();
};

/**
 * @class TimeSensitiveEvictionPolicy
 * @brief Time-sensitive eviction policy
 * @details
 *
 * Combines recency and frequency to make eviction decisions.
 * Also known as SLRU (Segmented Least Recently Used) or 2Q.
 * This is effective for mixed workload patterns.
 */
class TimeSensitiveEvictionPolicy : public IEvictionPolicy {
public:
  /**
   * @brief Constructor
   * @param recencyWeight Weight for recency vs frequency (0-1)
   */
  TimeSensitiveEvictionPolicy(double recencyWeight = 0.5);

  /**
   * @brief Notify the policy that an item was accessed
   * @param item The item that was accessed
   */
  void onAccess(void *item) override;

  /**
   * @brief Select an item for eviction
   * @return Pointer to the item to evict
   */
  void *selectForEviction() override;

  /**
   * @brief Add an item to the policy
   * @param item The item to add
   */
  void addItem(void *item) override;

  /**
   * @brief Remove an item from the policy
   * @param item The item to remove
   */
  void removeItem(void *item) override;

  /**
   * @brief Clear all items from the policy
   */
  void clear() override;

  /**
   * @brief Get the number of items in the policy
   * @return Number of items
   */
  size_t size() const override;

  /**
   * @brief Get the name of this policy
   * @return The policy name
   */
  std::string getName() const override;

  /**
   * @brief Get statistics for this policy
   * @return Map of statistic name to value
   */
  std::unordered_map<std::string, double> getStatistics() const override;

private:
  std::unordered_map<void *, double> m_scoreMap;
  std::unordered_map<void *, size_t> m_lastAccessMap;
  std::unordered_map<void *, size_t> m_accessCountMap;
  std::vector<std::pair<void *, double>> m_scoreList;
  size_t m_accessCount = 0;
  size_t m_totalAccessCount = 0;
  double m_recencyWeight;

  /**
   * @brief Calculate the score for an item
   * @param item The item to calculate score for
   * @return The score (lower means higher eviction priority)
   */
  double calculateScore(void *item);

  /**
   * @brief Update the score for an item
   * @param item The item to update
   */
  void updateScore(void *item);

  /**
   * @brief Sort the score list
   */
  void sortScoreList();
};

/**
 * @class AdaptiveEvictionPolicy
 * @brief Self-tuning eviction policy that adjusts to access patterns
 * @details
 *
 * The AdaptiveEvictionPolicy class implements a sophisticated cache eviction
 * strategy that automatically selects between multiple underlying policies
 * based on observed access patterns. This approach combines the strengths of
 * different eviction strategies to optimize cache performance for changing
 * workloads.
 *
 * Key features:
 * 1. Pattern detection: Analyzes access history to identify sequential scans,
 * loops, etc.
 * 2. Automatic policy selection: Switches between LRU, LFU, and time-sensitive
 * policies
 * 3. Multiple policy tracking: Maintains state for all policies simultaneously
 * 4. Configurable adaptation interval: Controls how frequently to re-evaluate
 * policy choice
 */
class AdaptiveEvictionPolicy : public IEvictionPolicy {
public:
  /**
   * @brief Constructor
   * @param adaptationInterval Number of accesses between adaptations
   */
  AdaptiveEvictionPolicy(size_t adaptationInterval = 1000);

  /**
   * @brief Notify the policy that an item was accessed
   * @param item The item that was accessed
   */
  void onAccess(void *item) override;

  /**
   * @brief Select an item for eviction
   * @return Pointer to the item to evict
   */
  void *selectForEviction() override;

  /**
   * @brief Add an item to the policy
   * @param item The item to add
   */
  void addItem(void *item) override;

  /**
   * @brief Remove an item from the policy
   * @param item The item to remove
   */
  void removeItem(void *item) override;

  /**
   * @brief Clear all items from the policy
   */
  void clear() override;

  /**
   * @brief Get the number of items in the policy
   * @return Number of items
   */
  size_t size() const override;

  /**
   * @brief Get the name of this policy
   * @return The policy name
   */
  std::string getName() const override;

  /**
   * @brief Get statistics for this policy
   * @return Map of statistic name to value
   */
  std::unordered_map<std::string, double> getStatistics() const override;

private:
  std::unique_ptr<LRUEvictionPolicy> m_lruPolicy;
  std::unique_ptr<FrequencyEvictionPolicy> m_lfuPolicy;
  std::unique_ptr<TimeSensitiveEvictionPolicy> m_timePolicy;

  IEvictionPolicy *m_activePolicy;
  std::string m_activePolicyName;

  size_t m_adaptationInterval;
  size_t m_accessCount = 0;

  // Access pattern tracking
  struct AccessInfo {
    std::vector<void *> recentAccesses;
    std::unordered_map<void *, size_t> accessCounts;
  };

  mutable AccessInfo m_accessInfo;

  /**
   * @brief Record an access for pattern analysis
   * @param item The item that was accessed
   */
  void recordAccess(void *item);

  /**
   * @brief Adapt the policy based on observed patterns
   */
  void adaptPolicy();

  /**
   * @brief Detect the dominant access pattern
   * @return String indicating the pattern type
   */
  std::string detectPattern() const;
};

