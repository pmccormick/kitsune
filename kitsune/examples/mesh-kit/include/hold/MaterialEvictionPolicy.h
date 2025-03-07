#pragma once

#include "MaterialCacheCore.h"
#include <chrono>
#include <list>
#include <mutex>
#include <random>
#include <string>
#include <unordered_map>

/**
 * @class LRUEvictionPolicy
 * @brief Least Recently Used eviction policy
 *
 * Evicts the least recently accessed items first.
 */
class LRUEvictionPolicy : public IEvictionPolicy {
public:
  void onAccess(void *item) override {
    auto it = m_itemMap.find(item);
    if (it != m_itemMap.end()) {
      // Move to front of list (most recently used)
      m_lruList.erase(it->second);
      m_lruList.push_front(item);
      it->second = m_lruList.begin();
      m_accessCount++;
    }
  }

  void *selectForEviction() override {
    if (!m_lruList.empty()) {
      return m_lruList.back(); // Return least recently used
    }
    return nullptr;
  }

  void addItem(void *item) override {
    if (m_itemMap.find(item) == m_itemMap.end()) {
      // Add to front of list (most recently used)
      m_lruList.push_front(item);
      m_itemMap[item] = m_lruList.begin();
    }
  }

  void removeItem(void *item) override {
    auto it = m_itemMap.find(item);
    if (it != m_itemMap.end()) {
      m_lruList.erase(it->second);
      m_itemMap.erase(it);
    }
  }

  void clear() override {
    m_lruList.clear();
    m_itemMap.clear();
    m_accessCount = 0;
  }

  size_t size() const override { return m_lruList.size(); }

  std::string getName() const override { return "LRU (Least Recently Used)"; }

  std::unordered_map<std::string, double> getStatistics() const override {
    return {{"Access Count", static_cast<double>(m_accessCount)},
            {"Item Count", static_cast<double>(m_lruList.size())}};
  }

private:
  std::list<void *> m_lruList;
  std::unordered_map<void *, std::list<void *>::iterator> m_itemMap;
  size_t m_accessCount = 0;
};

/**
 * @class MRUEvictionPolicy
 * @brief Most Recently Used eviction policy
 *
 * Evicts the most recently accessed items first.
 * This can be useful for scenarios with cyclic access patterns.
 */
class MRUEvictionPolicy : public IEvictionPolicy {
public:
  void onAccess(void *item) override {
    auto it = m_itemMap.find(item);
    if (it != m_itemMap.end()) {
      // Move to front of list (most recently used)
      m_mruList.erase(it->second);
      m_mruList.push_front(item);
      it->second = m_mruList.begin();
      m_accessCount++;
    }
  }

  void *selectForEviction() override {
    if (!m_mruList.empty()) {
      return m_mruList.front(); // Return most recently used
    }
    return nullptr;
  }

  void addItem(void *item) override {
    if (m_itemMap.find(item) == m_itemMap.end()) {
      // Add to front of list (most recently used)
      m_mruList.push_front(item);
      m_itemMap[item] = m_mruList.begin();
    }
  }

  void removeItem(void *item) override {
    auto it = m_itemMap.find(item);
    if (it != m_itemMap.end()) {
      m_mruList.erase(it->second);
      m_itemMap.erase(it);
    }
  }

  void clear() override {
    m_mruList.clear();
    m_itemMap.clear();
    m_accessCount = 0;
  }

  size_t size() const override { return m_mruList.size(); }

  std::string getName() const override { return "MRU (Most Recently Used)"; }

  std::unordered_map<std::string, double> getStatistics() const override {
    return {{"Access Count", static_cast<double>(m_accessCount)},
            {"Item Count", static_cast<double>(m_mruList.size())}};
  }

private:
  std::list<void *> m_mruList;
  std::unordered_map<void *, std::list<void *>::iterator> m_itemMap;
  size_t m_accessCount = 0;
};

/**
 * @class FIFOEvictionPolicy
 * @brief First In First Out eviction policy
 *
 * Evicts items in the order they were added, regardless of access patterns.
 */
class FIFOEvictionPolicy : public IEvictionPolicy {
public:
  void onAccess(void *item) override {
    // No change in order, just record the access
    if (m_itemSet.find(item) != m_itemSet.end()) {
      m_accessCount++;
    }
  }

  void *selectForEviction() override {
    if (!m_fifoQueue.empty()) {
      return m_fifoQueue.back(); // Return oldest inserted
    }
    return nullptr;
  }

  void addItem(void *item) override {
    if (m_itemSet.find(item) == m_itemSet.end()) {
      // Add to front of queue (newest)
      m_fifoQueue.push_front(item);
      m_itemSet.insert(item);
    }
  }

  void removeItem(void *item) override {
    auto it = std::find(m_fifoQueue.begin(), m_fifoQueue.end(), item);
    if (it != m_fifoQueue.end()) {
      m_fifoQueue.erase(it);
      m_itemSet.erase(item);
    }
  }

  void clear() override {
    m_fifoQueue.clear();
    m_itemSet.clear();
    m_accessCount = 0;
  }

  size_t size() const override { return m_fifoQueue.size(); }

  std::string getName() const override { return "FIFO (First In First Out)"; }

  std::unordered_map<std::string, double> getStatistics() const override {
    return {{"Access Count", static_cast<double>(m_accessCount)},
            {"Item Count", static_cast<double>(m_fifoQueue.size())}};
  }

private:
  std::list<void *> m_fifoQueue;
  std::unordered_set<void *> m_itemSet; // For fast existence check
  size_t m_accessCount = 0;
};

/**
 * @class FrequencyEvictionPolicy
 * @brief Frequency-based eviction policy
 *
 * Evicts the least frequently accessed items first (LFU).
 * This is useful for workloads with stable popularity patterns.
 */
class FrequencyEvictionPolicy : public IEvictionPolicy {
public:
  void onAccess(void *item) override {
    auto it = m_frequencyMap.find(item);
    if (it != m_frequencyMap.end()) {
      // Increase frequency counter
      it->second++;
      m_accessCount++;

      // Update the frequency-ordered list
      updateFrequencyList(item);
    }
  }

  void *selectForEviction() override {
    if (!m_frequencyList.empty()) {
      // Return item with lowest frequency
      return m_frequencyList.front().first;
    }
    return nullptr;
  }

  void addItem(void *item) override {
    if (m_frequencyMap.find(item) == m_frequencyMap.end()) {
      // Initialize with frequency 1
      m_frequencyMap[item] = 1;

      // Add to frequency list
      m_frequencyList.push_back(std::make_pair(item, 1));

      // Sort the list by frequency (least frequent first)
      sortFrequencyList();
    }
  }

  void removeItem(void *item) override {
    auto it = m_frequencyMap.find(item);
    if (it != m_frequencyMap.end()) {
      m_frequencyMap.erase(it);

      // Remove from frequency list
      auto listIt =
          std::find_if(m_frequencyList.begin(), m_frequencyList.end(),
                       [item](const auto &pair) { return pair.first == item; });
      if (listIt != m_frequencyList.end()) {
        m_frequencyList.erase(listIt);
      }
    }
  }

  void clear() override {
    m_frequencyMap.clear();
    m_frequencyList.clear();
    m_accessCount = 0;
  }

  size_t size() const override { return m_frequencyMap.size(); }

  std::string getName() const override { return "LFU (Least Frequently Used)"; }

  std::unordered_map<std::string, double> getStatistics() const override {
    double avgFrequency = 0.0;
    size_t maxFrequency = 0;

    if (!m_frequencyMap.empty()) {
      size_t totalFrequency = 0;
      for (const auto &pair : m_frequencyMap) {
        totalFrequency += pair.second;
        maxFrequency = std::max(maxFrequency, pair.second);
      }
      avgFrequency =
          static_cast<double>(totalFrequency) / m_frequencyMap.size();
    }

    return {{"Access Count", static_cast<double>(m_accessCount)},
            {"Item Count", static_cast<double>(m_frequencyMap.size())},
            {"Average Frequency", avgFrequency},
            {"Maximum Frequency", static_cast<double>(maxFrequency)}};
  }

private:
  std::unordered_map<void *, size_t> m_frequencyMap; // Item to frequency count
  std::vector<std::pair<void *, size_t>> m_frequencyList; // Sorted by frequency
  size_t m_accessCount = 0;

  void updateFrequencyList(void *item) {
    // Find and update the item in the list
    auto it =
        std::find_if(m_frequencyList.begin(), m_frequencyList.end(),
                     [item](const auto &pair) { return pair.first == item; });

    if (it != m_frequencyList.end()) {
      it->second = m_frequencyMap[item];
      sortFrequencyList();
    }
  }

  void sortFrequencyList() {
    // Sort by frequency (ascending)
    std::sort(m_frequencyList.begin(), m_frequencyList.end(),
              [](const auto &a, const auto &b) { return a.second < b.second; });
  }
};

/**
 * @class TimeSensitiveEvictionPolicy
 * @brief Time-sensitive eviction policy
 *
 * Combines recency and frequency to make eviction decisions.
 * Also known as SLRU (Segmented Least Recently Used) or 2Q.
 * This is effective for mixed workload patterns.
 */
class TimeSensitiveEvictionPolicy : public IEvictionPolicy {
public:
  TimeSensitiveEvictionPolicy(double recencyWeight = 0.5)
      : m_recencyWeight(recencyWeight) {}

  void onAccess(void *item) override {
    auto it = m_scoreMap.find(item);
    if (it != m_scoreMap.end()) {
      // Update last access time
      m_lastAccessMap[item] = m_accessCount;

      // Increase access count
      m_accessCountMap[item]++;

      // Update score
      updateScore(item);

      m_totalAccessCount++;
    }
  }

  void *selectForEviction() override {
    if (m_scoreList.empty()) {
      return nullptr;
    }

    // Return item with lowest score
    return m_scoreList.front().first;
  }

  void addItem(void *item) override {
    if (m_scoreMap.find(item) == m_scoreMap.end()) {
      // Initialize
      m_lastAccessMap[item] = m_accessCount;
      m_accessCountMap[item] = 1;

      // Calculate initial score
      double score = calculateScore(item);
      m_scoreMap[item] = score;

      // Add to score list
      m_scoreList.push_back(std::make_pair(item, score));

      // Sort the list by score (lowest first)
      sortScoreList();

      m_accessCount++;
    }
  }

  void removeItem(void *item) override {
    auto it = m_scoreMap.find(item);
    if (it != m_scoreMap.end()) {
      m_scoreMap.erase(it);
      m_lastAccessMap.erase(item);
      m_accessCountMap.erase(item);

      // Remove from score list
      auto listIt =
          std::find_if(m_scoreList.begin(), m_scoreList.end(),
                       [item](const auto &pair) { return pair.first == item; });
      if (listIt != m_scoreList.end()) {
        m_scoreList.erase(listIt);
      }
    }
  }

  void clear() override {
    m_scoreMap.clear();
    m_lastAccessMap.clear();
    m_accessCountMap.clear();
    m_scoreList.clear();
    m_accessCount = 0;
    m_totalAccessCount = 0;
  }

  size_t size() const override { return m_scoreMap.size(); }

  std::string getName() const override {
    return "Time-Sensitive (Recency+Frequency)";
  }

  std::unordered_map<std::string, double> getStatistics() const override {
    return {{"Access Count", static_cast<double>(m_totalAccessCount)},
            {"Item Count", static_cast<double>(m_scoreMap.size())},
            {"Recency Weight", m_recencyWeight}};
  }

private:
  std::unordered_map<void *, double> m_scoreMap; // Item to score
  std::unordered_map<void *, size_t>
      m_lastAccessMap; // Item to last access time
  std::unordered_map<void *, size_t> m_accessCountMap; // Item to access count
  std::vector<std::pair<void *, double>> m_scoreList;  // Sorted by score
  size_t m_accessCount = 0;      // Counter for access ordering
  size_t m_totalAccessCount = 0; // Total number of accesses
  double m_recencyWeight;        // Weight for recency vs frequency (0-1)

  double calculateScore(void *item) {
    // Normalize access counts to 0-1 range
    double maxAccessCount = 1.0;
    for (const auto &pair : m_accessCountMap) {
      maxAccessCount =
          std::max(maxAccessCount, static_cast<double>(pair.second));
    }

    // Calculate normalized frequency score (higher is better)
    double frequencyScore =
        static_cast<double>(m_accessCountMap[item]) / maxAccessCount;

    // Calculate normalized recency score (higher is better)
    double recencyScore =
        static_cast<double>(m_lastAccessMap[item]) / m_accessCount;

    // Combine scores (invert for eviction priority - lower is evicted first)
    return 1.0 - (m_recencyWeight * recencyScore +
                  (1.0 - m_recencyWeight) * frequencyScore);
  }

  void updateScore(void *item) {
    // Recalculate score
    double score = calculateScore(item);
    m_scoreMap[item] = score;

    // Update in score list
    auto listIt =
        std::find_if(m_scoreList.begin(), m_scoreList.end(),
                     [item](const auto &pair) { return pair.first == item; });
    if (listIt != m_scoreList.end()) {
      listIt->second = score;
      sortScoreList();
    }
  }

  void sortScoreList() {
    // Sort by score (ascending - lower scores evicted first)
    std::sort(m_scoreList.begin(), m_scoreList.end(),
              [](const auto &a, const auto &b) { return a.second < b.second; });
  }
};

/**
 * @class AdaptiveEvictionPolicy
 * @brief Adaptive eviction policy
 *
 * Automatically selects between LRU, LFU, and other policies
 * based on observed access patterns.
 */
class AdaptiveEvictionPolicy : public IEvictionPolicy {
public:
  AdaptiveEvictionPolicy(size_t adaptationInterval = 1000)
      : m_adaptationInterval(adaptationInterval) {
    // Create underlying policies
    m_lruPolicy = std::make_unique<LRUEvictionPolicy>();
    m_lfuPolicy = std::make_unique<FrequencyEvictionPolicy>();
    m_timePolicy = std::make_unique<TimeSensitiveEvictionPolicy>();

    // Start with LRU
    m_activePolicy = m_lruPolicy.get();
    m_activePolicyName = "LRU";
  }

  void onAccess(void *item) override {
    // Forward to all policies
    m_lruPolicy->onAccess(item);
    m_lfuPolicy->onAccess(item);
    m_timePolicy->onAccess(item);

    // Record access pattern
    recordAccess(item);

    // Check if adaptation is needed
    m_accessCount++;
    if (m_accessCount % m_adaptationInterval == 0) {
      adaptPolicy();
    }
  }

  void *selectForEviction() override {
    // Use active policy
    return m_activePolicy->selectForEviction();
  }

  void addItem(void *item) override {
    // Add to all policies
    m_lruPolicy->addItem(item);
    m_lfuPolicy->addItem(item);
    m_timePolicy->addItem(item);

    // Initialize access record
    m_accessRecord[item] = std::list<bool>();
  }

  void removeItem(void *item) override {
    // Remove from all policies
    m_lruPolicy->removeItem(item);
    m_lfuPolicy->removeItem(item);
    m_timePolicy->removeItem(item);

    // Remove access record
    m_accessRecord.erase(item);
  }

  void clear() override {
    // Clear all policies
    m_lruPolicy->clear();
    m_lfuPolicy->clear();
    m_timePolicy->clear();

    // Reset state
    m_accessRecord.clear();
    m_accessCount = 0;
    m_scanCount = 0;
    m_loopCount = 0;
  }

  size_t size() const override { return m_lruPolicy->size(); }

  std::string getName() const override {
    return "Adaptive (" + m_activePolicyName + ")";
  }