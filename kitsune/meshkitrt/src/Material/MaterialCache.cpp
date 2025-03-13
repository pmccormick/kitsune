/**
 * @file MaterialCache.cpp
 * @brief Implementation of the material cache base class for scientific
 * simulations
 *
 * This file implements the methods declared in MaterialCache.h, providing
 * the core functionality for material caching with equivalence checking in
 * computational science applications. The implementation focuses on thread
 * safety, efficient lookups, and proper interaction with the pluggable eviction
 * and analytics policies.
 *
 * Performance considerations for scientific computing:
 * - Lock granularity is carefully managed to minimize thread contention
 * - Hash-based lookup is optimized for computational materials science
 * - Memory usage is controlled through strategic eviction
 * - Analytics hooks provide minimal overhead when enabled
 */

#include "MaterialCache.h"
#include "LRUEvictionPolicy.h"

MaterialCacheBase::MaterialCacheBase(
    size_t maxSize, std::unique_ptr<MaterialEquivalence> equivalence,
    std::unique_ptr<CacheEvictionPolicy> evictionPolicy,
    std::unique_ptr<CacheAnalyticsPolicy> analyticsPolicy)
    : m_maxSize(maxSize), m_equivalence(std::move(equivalence)),
      m_evictionPolicy(std::move(evictionPolicy)),
      m_analyticsPolicy(std::move(analyticsPolicy)),
      m_cache(10, MaterialHasher{m_equivalence.get(), m_analyticsPolicy.get()},
              MaterialEqual{m_equivalence.get(), m_analyticsPolicy.get()}) {
  // If no eviction policy provided, use LRU by default
  // LRU is a good general-purpose choice for scientific simulations
  // as it works well with the temporal locality typical in iterative solvers
  if (!m_evictionPolicy) {
    m_evictionPolicy = std::make_unique<LRUEvictionPolicy>();
  }
}

std::shared_ptr<Material>
MaterialCacheBase::findOrAdd(const std::shared_ptr<Material> &material) {
  // First try to find without adding
  auto found = find(material);
  if (found) {
    return found;
  }

  // Not found, add to cache
  add(material);
  return material;
}

std::shared_ptr<Material>
MaterialCacheBase::find(const std::shared_ptr<Material> &material) {
  // Thread-safe lookup with minimal lock duration
  std::lock_guard<std::mutex> lock(m_mutex);

  auto it = m_cache.find(material);
  if (it != m_cache.end()) {
    // Found in cache - update eviction policy tracking
    if (m_evictionPolicy) {
      m_evictionPolicy->onAccess(it->second.get(), this);
    }

    // Record hit in analytics if available
    if (m_analyticsPolicy) {
      m_analyticsPolicy->onCacheHit(it->second.get());
    }

    return it->second;
  }

  // Not found - record miss in analytics
  if (m_analyticsPolicy) {
    m_analyticsPolicy->onCacheMiss(material);
  }

  return nullptr;
}

void MaterialCacheBase::add(const std::shared_ptr<Material> &material) {
  std::lock_guard<std::mutex> lock(m_mutex);

  // Check if already exists - common in scientific simulations
  // where same material may be created multiple times
  auto it = m_cache.find(material);
  if (it != m_cache.end()) {
    // Already in cache, just update access tracking
    if (m_evictionPolicy) {
      m_evictionPolicy->onAccess(it->second.get(), this);
    }

    // Record hit in analytics
    if (m_analyticsPolicy) {
      m_analyticsPolicy->onCacheHit(it->second.get());
    }

    return;
  }

  // Check if we need to make room by evicting
  evictIfNeeded();

  // Add the material to cache
  m_cache[material] = material;

  // Notify eviction policy
  if (m_evictionPolicy) {
    m_evictionPolicy->onAdd(material.get(), this);
  }

  // Record addition in analytics
  if (m_analyticsPolicy) {
    m_analyticsPolicy->onItemAdded(material.get());
  }
}

size_t MaterialCacheBase::size() const {
  std::lock_guard<std::mutex> lock(m_mutex);
  return m_cache.size();
}

void MaterialCacheBase::clear() {
  std::lock_guard<std::mutex> lock(m_mutex);
  m_cache.clear();
}

void MaterialCacheBase::setMaxSize(size_t size) {
  std::lock_guard<std::mutex> lock(m_mutex);
  m_maxSize = size;

  // If cache is now too big, evict items until within new limit
  while (m_maxSize > 0 && m_cache.size() > m_maxSize) {
    evictIfNeeded();
  }
}

size_t MaterialCacheBase::getMaxSize() const { return m_maxSize; }

const MaterialEquivalence &MaterialCacheBase::getEquivalence() const {
  return *m_equivalence;
}

void MaterialCacheBase::setEvictionPolicy(
    std::unique_ptr<CacheEvictionPolicy> policy) {
  std::lock_guard<std::mutex> lock(m_mutex);
  m_evictionPolicy = std::move(policy);

  // Rebuild policy data by adding all items to the new policy
  // This ensures the eviction policy has a complete view of the cache
  if (m_evictionPolicy) {
    for (const auto &pair : m_cache) {
      m_evictionPolicy->onAdd(pair.second.get(), this);
    }
  }
}

void MaterialCacheBase::setAnalyticsPolicy(
    std::unique_ptr<CacheAnalyticsPolicy> policy) {
  std::lock_guard<std::mutex> lock(m_mutex);
  m_analyticsPolicy = std::move(policy);

  // Need to rebuild the cache to update the hasher and equality comparator
  // which may depend on the analytics policy for instrumentation
  rebuildCache();
}

const CacheAnalyticsPolicy *MaterialCacheBase::getAnalyticsPolicy() const {
  return m_analyticsPolicy.get();
}

std::unordered_map<std::string, double>
MaterialCacheBase::getStatistics() const {
  std::lock_guard<std::mutex> lock(m_mutex);

  if (m_analyticsPolicy) {
    return m_analyticsPolicy->getStatistics();
  }

  // Return basic stats if no analytics policy is set
  // This ensures minimal functionality even without explicit analytics
  return {{"Cache Size", static_cast<double>(m_cache.size())},
          {"Max Size", static_cast<double>(m_maxSize)}};
}

void MaterialCacheBase::resetStatistics() {
  std::lock_guard<std::mutex> lock(m_mutex);

  if (m_analyticsPolicy) {
    m_analyticsPolicy->reset();
  }
}

void MaterialCacheBase::evictIfNeeded() {
  // This helper method is called when cache is at capacity
  // and before adding new items
  if (m_maxSize > 0 && m_cache.size() >= m_maxSize) {
    // If we have an eviction policy, use it to select victim
    if (m_evictionPolicy) {
      void *toEvict = m_evictionPolicy->selectVictim(this);

      if (toEvict) {
        // Find and remove the victim from cache
        // Linear search is acceptable here as eviction is relatively infrequent
        // compared to lookups in well-sized scientific caches
        for (auto it = m_cache.begin(); it != m_cache.end(); ++it) {
          if (it->second.get() == toEvict) {
            // Record eviction in analytics
            if (m_analyticsPolicy) {
              m_analyticsPolicy->onItemEvicted(toEvict);
            }

            m_cache.erase(it);
            break;
          }
        }
      }
    }
    // Fallback if no eviction policy or if policy didn't select a victim
    else if (!m_cache.empty()) {
      auto it = m_cache.begin();

      // Record eviction in analytics
      if (m_analyticsPolicy) {
        m_analyticsPolicy->onItemEvicted(it->second.get());
      }

      m_cache.erase(it);
    }
  }
}

void MaterialCacheBase::rebuildCache() {
  // This helper reconstructs the cache with updated configuration
  // Used primarily when changing analytics policies which affect the hash/equal
  // functors

  // Save all materials
  std::vector<std::shared_ptr<Material>> materials;
  materials.reserve(m_cache.size());

  for (const auto &pair : m_cache) {
    materials.push_back(pair.second);
  }

  // Create a new cache with updated hasher and equality comparator
  std::unordered_map<std::shared_ptr<Material>, std::shared_ptr<Material>,
                     MaterialHasher, MaterialEqual>
      newCache(materials.size() * 2, // Sizing with load factor for efficiency
               MaterialHasher{m_equivalence.get(), m_analyticsPolicy.get()},
               MaterialEqual{m_equivalence.get(), m_analyticsPolicy.get()});

  // Add all materials to the new cache and notify policies
  for (const auto &material : materials) {
    newCache[material] = material;

    if (m_evictionPolicy) {
      m_evictionPolicy->onAdd(material.get(), this);
    }

    if (m_analyticsPolicy) {
      m_analyticsPolicy->onItemAdded(material.get());
    }
  }

  // Swap with the old cache
  m_cache.swap(newCache);

  // Old cache is automatically destroyed when this method ends
  // freeing all its resources
}
