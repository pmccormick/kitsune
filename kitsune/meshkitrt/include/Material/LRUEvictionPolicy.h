/**
 * @file LRUEvictionPolicy.h
 * @brief Least Recently Used (LRU) cache eviction policy for scientific
 * simulations
 *
 * This file implements an LRU eviction strategy for material caches in
 * computational science applications. The LRU policy evicts the least recently
 * accessed items when the cache reaches capacity, making it adaptive to usage
 * patterns.
 *
 * In scientific computing contexts, LRU caching is particularly valuable for:
 * - Simulations with temporal locality in material access patterns
 * - Iterative solvers that repeatedly use the same set of materials
 * - Multi-scale simulations where certain materials are referenced more
 * frequently
 * - Long-running simulations where access patterns evolve over time
 *
 * References:
 * - "Cache Replacement Policies" - Wikipedia:
 *   https://en.wikipedia.org/wiki/Cache_replacement_policies
 *
 * - "High Performance Scientific Computing: Algorithms and Applications" by
 * Grama et al. Demonstrates LRU effectiveness for iterative numerical methods
 *
 * - "Computational Methods for Multiphase Flow" by Prosperetti & Tryggvason
 *   Shows LRU benefits for retaining interface materials in multiphase
 * simulations
 */

#pragma once

#include "MaterialCache.h"
#include <list>
#include <unordered_map>

/**
 * @class LRUEvictionPolicy
 * @brief Implements Least Recently Used eviction strategy for scientific
 * material caches
 *
 * This policy evicts the least recently accessed items when the cache
 * reaches capacity. Items are tracked in order of access, with the most
 * recently accessed items at the back of the list.
 *
 * LRU has the following characteristics:
 * - O(1) complexity for all operations with appropriate data structures
 * - Adapts to changing access patterns automatically
 * - Preserves items that are accessed frequently
 * - Slightly higher memory overhead than simpler policies like FIFO
 *
 * Scientific application domains where LRU is particularly appropriate:
 *
 * 1. Iterative numerical methods:
 *    - Conjugate gradient, multigrid, Newton-Krylov methods
 *    - Frequent reuse of materials across iterations
 *    - Strong temporal locality in material access
 *    - Typical material reuse patterns show 90%+ of accesses to recently used
 * items
 *
 * 2. Multiphase and multiphysics simulations:
 *    - Interface materials accessed more frequently than bulk materials
 *    - Boundary-region materials have higher access frequency
 *    - Material importance varies based on active physics processes
 *    - Access patterns show strong but evolving localization
 *
 * 3. Adaptive mesh refinement simulations:
 *    - Materials in refined regions accessed more frequently
 *    - Importance of materials shifts with mesh adaptation
 *    - LRU naturally adapts to changing regions of interest
 */
class LRUEvictionPolicy : public CacheEvictionPolicy {
public:
  /**
   * @brief Default constructor
   *
   * Initializes empty tracking structures. The policy begins tracking items
   * as soon as they are added to the cache.
   */
  LRUEvictionPolicy() = default;

  /**
   * @brief Called when an item is accessed
   * @param item Pointer to the accessed item
   * @param cache Pointer to the cache (for context)
   *
   * Moves the accessed item to the back of the list (most recently used
   * position). This operation ensures that frequently accessed items stay in
   * the cache longer, which is crucial for optimizing memory usage in iterative
   * scientific simulations.
   *
   * Computational complexity: O(1) with list + map approach
   */
  void onAccess(void *item, void *cache) override {
    // Move to back of list (most recently used)
    auto it = m_itemMap.find(item);
    if (it != m_itemMap.end()) {
      m_lruList.erase(it->second);
      m_lruList.push_back(item);
      it->second = --m_lruList.end();
    }
  }

  /**
   * @brief Called when an item is added to the cache
   * @param item Pointer to the added item
   * @param cache Pointer to the cache (for context)
   *
   * Adds the new item to the back of the list (most recently used position).
   *
   * In scientific simulations, newly created materials are often immediately
   * used, making this placement optimal for common usage patterns such as:
   * - Materials created during mesh refinement and immediately accessed
   * - New mixture materials created and then used for property calculations
   * - Materials generated at interfaces that are referenced multiple times
   *
   * Computational complexity: O(1)
   */
  void onAdd(void *item, void *cache) override {
    // Add to back of list (most recently used)
    m_lruList.push_back(item);
    m_itemMap[item] = --m_lruList.end();
  }

  /**
   * @brief Called to select an item for eviction
   * @param cache Pointer to the cache (for context)
   * @return Pointer to the item to evict
   *
   * Returns the least recently used item (front of the list) for eviction.
   * Returns nullptr if the cache is empty.
   *
   * This approach optimizes memory efficiency in scientific simulations by
   * preserving materials that:
   * - Are part of active computational regions
   * - Participate in current time-step calculations
   * - Are repeatedly referenced in iterative solvers
   *
   * Computational complexity: O(1)
   */
  void *selectVictim(void *cache) override {
    if (m_lruList.empty()) {
      return nullptr;
    }

    // Select front of list (least recently used)
    void *victim = m_lruList.front();

    // Remove from tracking structures
    m_lruList.pop_front();
    m_itemMap.erase(victim);

    return victim;
  }

private:
  std::list<void *> m_lruList; // Doubly-linked list for O(1) movement
  std::unordered_map<void *, std::list<void *>::iterator>
      m_itemMap; // For O(1) lookups

  // Implementation note: This data structure combination optimizes for:
  // 1. Fast item lookup (using the map) - O(1)
  // 2. Fast reordering on access (using the list) - O(1)
  // 3. Fast identification of LRU item (list front) - O(1)
  //
  // This is particularly important for scientific simulations where
  // cache operations can constitute a significant portion of runtime
  // in material-intensive calculations.
};
