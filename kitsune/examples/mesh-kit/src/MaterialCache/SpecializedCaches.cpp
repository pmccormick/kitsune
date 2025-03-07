/**
 * @file SpecializedCaches.cpp
 * @brief Implementation of specialized material cache classes
 */

#include "SpecializedCaches.h"

//==============================================================================
// FixedSizeMaterialCache Implementation
//==============================================================================

FixedSizeMaterialCache::FixedSizeMaterialCache(size_t size, double tolerance)
    : MaterialCache(size, tolerance) {}

FixedSizeMaterialCache::FixedSizeMaterialCache(
    size_t size, std::unique_ptr<IMaterialEquivalenceKey> equivalenceKey,
    std::unique_ptr<IEvictionPolicy> evictionPolicy)
    : MaterialCache(std::move(equivalenceKey), std::move(evictionPolicy),
                    size) {}

void FixedSizeMaterialCache::setMaxSize([[maybe_unused]] size_t size) {
  // Cannot change size of fixed cache
  // Silently ignore the request
}

std::string FixedSizeMaterialCache::getName() const {
  return "FixedSizeMaterialCache [Size: " + std::to_string(getMaxSize()) + "]";
}

//==============================================================================
// AdaptiveMaterialCache Implementation
//==============================================================================

AdaptiveMaterialCache::AdaptiveMaterialCache(size_t initialSize, size_t maxSize,
                                             double initialTolerance,
                                             size_t adaptationInterval)
    : MaterialCache(
          std::make_unique<AdaptiveEquivalenceKey>(initialTolerance,
                                                   adaptationInterval),
          std::make_unique<AdaptiveEvictionPolicy>(adaptationInterval),
          initialSize, true),
      m_initialSize(initialSize), m_maxSize(maxSize),
      m_adaptationInterval(adaptationInterval), m_lastSizeAdaptation(0),
      m_lastConfigAdaptation(0) {}

          std::shared_ptr<Material> AdaptiveMaterialCache::getOrAdd(
              const std::shared_ptr<Material> &material) {
  // Call parent implementation
  auto result = MaterialCache::getOrAdd(material);

  // Check if adaptation is needed
  m_operationsSinceLastAdaptation++;
  if (m_operationsSinceLastAdaptation >= m_adaptationInterval) {
    adaptCache();
    m_operationsSinceLastAdaptation = 0;
  }

  return result;
}

std::string AdaptiveMaterialCache::getName() const {
  return "AdaptiveMaterialCache [Size: " + std::to_string(size()) + "/" +
         std::to_string(getMaxSize()) +
         ", Adaptations: " + std::to_string(m_adaptationCount) + "]";
}

std::unordered_map<std::string, double>
AdaptiveMaterialCache::getAdaptiveStatistics() const {
  return {{"Adaptation Count", static_cast<double>(m_adaptationCount)},
          {"Size Adaptations", static_cast<double>(m_sizeAdaptationCount)},
          {"Config Adaptations", static_cast<double>(m_configAdaptationCount)},
          {"Initial Size", static_cast<double>(m_initialSize)},
          {"Maximum Size", static_cast<double>(m_maxSize)},
          {"Adaptation Interval", static_cast<double>(m_adaptationInterval)}};
}

void AdaptiveMaterialCache::adaptCache() {
  std::lock_guard<std::mutex> lock(m_mutex);

  m_adaptationCount++;

  // Record current hit rate
  const double hitRate = getStatistics().getHitRate();
  m_hitRateHistory.push_back(hitRate);
  if (m_hitRateHistory.size() > MAX_HISTORY) {
    m_hitRateHistory.erase(m_hitRateHistory.begin());
  }

  // Adapt cache size
  adaptCacheSize();

  // Adapt cache configuration
  adaptCacheConfiguration();
}

void AdaptiveMaterialCache::adaptCacheSize() {
  // Only adapt size after collecting enough history
  if (m_hitRateHistory.size() < 3) {
    return;
  }

  // Calculate average hit rate
  double avgHitRate = 0.0;
  for (double rate : m_hitRateHistory) {
    avgHitRate += rate;
  }
  avgHitRate /= m_hitRateHistory.size();

  // Get current size
  size_t currentSize = getMaxSize();
  size_t newSize = currentSize;

  // Adjust size based on hit rate
  if (avgHitRate < 20.0) {
    // Very low hit rate - increase size significantly
    newSize = std::min(currentSize * 2, m_maxSize);
  } else if (avgHitRate < 50.0) {
    // Moderate hit rate - increase size moderately
    newSize = std::min(currentSize * 3 / 2, m_maxSize);
  } else if (avgHitRate > 95.0) {
    // Very high hit rate - decrease size to save memory
    newSize = std::max(currentSize * 2 / 3, m_initialSize);
  } else if (avgHitRate > 80.0) {
    // High hit rate - slight decrease
    newSize = std::max(currentSize * 4 / 5, m_initialSize);
  }

  // Apply new size if changed
  if (newSize != currentSize) {
    MaterialCache::setMaxSize(newSize);
    m_lastSizeAdaptation = m_adaptationCount;
    m_sizeAdaptationCount++;
  }
}

void AdaptiveMaterialCache::adaptCacheConfiguration() {
  // Adapt less frequently than size
  if (m_adaptationCount - m_lastConfigAdaptation < 5) {
    return;
  }

  // Get current hit rate trend (increasing, decreasing, or stable)
  double hitRateTrend = 0.0;
  if (m_hitRateHistory.size() >= 5) {
    double recentAvg = 0.0;
    double olderAvg = 0.0;

    // Average of 3 most recent rates
    for (size_t i = m_hitRateHistory.size() - 3; i < m_hitRateHistory.size();
         i++) {
      recentAvg += m_hitRateHistory[i];
    }
    recentAvg /= 3.0;

    // Average of 3 older rates
    for (size_t i = m_hitRateHistory.size() - 6;
         i < m_hitRateHistory.size() - 3; i++) {
      olderAvg += m_hitRateHistory[i];
    }
    olderAvg /= 3.0;

    hitRateTrend = recentAvg - olderAvg;
  }

  // Get current configuration
  auto *adaptiveKey =
      dynamic_cast<AdaptiveEquivalenceKey *>(m_equivalenceKey.get());
  //auto *adaptivePolicy =
  //      dynamic_cast<AdaptiveEvictionPolicy *>(m_evictionPolicy.get());

  // If we're using an adaptive equivalence key, update its tolerance based on
  // the trend
  if (adaptiveKey) {
    double currentTolerance = adaptiveKey->getTolerance();
    double newTolerance = currentTolerance;

    if (hitRateTrend < -10.0) {
      // Hit rate decreasing - try more lenient tolerance
      newTolerance = currentTolerance * 2.0;
    } else if (hitRateTrend > 10.0) {
      // Hit rate increasing - maybe try more strict tolerance
      newTolerance = currentTolerance * 0.8;
    }

    // Don't change tolerance too drastically
    newTolerance = std::max(1e-12, std::min(1e-2, newTolerance));

    // Apply changes if significant
    if (std::abs(newTolerance - currentTolerance) / currentTolerance > 0.1) {
      adaptiveKey->setTolerance(newTolerance);
      m_lastConfigAdaptation = m_adaptationCount;
      m_configAdaptationCount++;
    }
  }
}

