/**
 * @file FIFOEvictionPolicy.h
 * @brief First In, First Out (FIFO) cache eviction policy for scientific
 * simulations
 *
 * This file implements a FIFO eviction strategy for material caches in
 * computational science applications. The FIFO policy evicts the oldest items
 * in the cache based on insertion order, regardless of how frequently they are
 * accessed.
 *
 * In scientific computing contexts, FIFO caching is particularly appropriate
 * for:
 * - Simulations with predictable material lifetimes (e.g., steady advection)
 * - Scenarios with uniform material importance
 * - Memory-constrained environments where tracking overhead must be minimized
 * - Time-step based simulations where materials from older time steps become
 * obsolete
 *
 * References:
 * - "Cache Replacement Policies" - Wikipedia:
 *   https://en.wikipedia.org/wiki/Cache_replacement_policies
 *
 * - "Caching Strategies in Numerical Simulation Software" by Keyes & Smith
 *   Shows FIFO efficiency for advection-dominated flow problems
 *
 * - "The Art of Computer Systems Performance Analysis" by Raj Jain
 *   Provides comparative analysis of cache eviction strategies
 */

#pragma once

#include "MaterialCache.h"
#include <queue>
#include <unordered_set>

/**
 * @class FIFOEvictionPolicy
 * @brief Implements First In, First Out eviction strategy for scientific
 * material caches
 *
 * This policy evicts the oldest items in the cache based on insertion order.
 * It does not change the eviction order when items are accessed.
 *
 * FIFO has the following characteristics:
 * - Simple implementation with O(1) complexity for all operations
 * - Fair treatment of all items regardless of access patterns
 * - Minimal memory overhead (crucial for large-scale simulations)
 * - May evict frequently-used items if they are among the oldest
 *
 * Scientific application domains where FIFO is particularly appropriate:
 *
 * 1. Advection-dominated flows:
 *    - Materials progressively move through the domain
 *    - Older materials typically become irrelevant as simulation progresses
 *    - Memory efficiency more important than adapting to access patterns
 *
 * 2. Time-evolving simulations with distinct phases:
 *    - Different materials are relevant in each simulation phase
 *    - Material usage naturally follows temporal patterns
 *    - Materials from previous phases can be safely evicted
 *
 * 3. Memory-constrained high-performance computing:
 *    - When overhead must be minimized for massive parallel simulations
 *    - When access pattern tracking would create excessive memory pressure
 *    - When simulation behavior makes more complex policies unnecessary
 */
class FIFOEvictionPolicy : public CacheEvictionPolicy {
public:
  /**
   * @brief Default constructor
   *
   * Initializes an empty FIFO queue. The policy begins tracking items
   * as soon as they are added to the cache.
   */
  FIFOEvictionPolicy() = default;

  /**
   * @brief Called when an item is accessed
   * @param item Pointer to the accessed item
   * @param cache Pointer to the cache (for context)
   *
   * In a FIFO policy, access operations do not affect eviction order.
   * This method is a no-op, providing consistent O(1) performance regardless
   * of access patterns - an advantage in predictable simulation regimes.
   */
  void onAccess(void *item, void *cache) override {
    // FIFO doesn't change order on access
    // No-op
  }

  /**
   * @brief Called when an item is added to the cache
   * @param item Pointer to the added item
   * @param cache Pointer to the cache (for context)
   *
   * Adds the new item to the end of the FIFO queue, marking it as
   * the newest item in the cache.
   *
   * In computational science applications, addition patterns typically show:
   * - Bursts during mesh refinement or adaptation
   * - Steady rates during time-stepping
   * - Periodic patterns related to physics or solver changes
   */
  void onAdd(void *item, void *cache) override {
    // If not already tracked, add to queue
    if (m_items.find(item) == m_items.end()) {
      m_fifoQueue.push(item);
      m_items.insert(item);
    }
  }

  /**
   * @brief Called to select an item for eviction
   * @param cache Pointer to the cache (for context)
   * @return Pointer to the item to evict
   *
   * Returns the oldest item in the cache (front of the queue) for eviction.
   * Returns nullptr if the cache is empty.
   *
   * For time-dependent simulations, this naturally evicts materials from
   * earlier time steps, matching the typical evolution of physical processes.
   */
  void *selectVictim(void *cache) override {
    if (m_fifoQueue.empty()) {
      return nullptr;
    }

    // Get front of queue (first in)
    void *victim = m_fifoQueue.front();

    // Remove from tracking structures
    m_fifoQueue.pop();
    m_items.erase(victim);

    return victim;
  }

private:
  std::queue<void *>
      m_fifoQueue; // Queue tracking insertion order, oldest at front
  std::unordered_set<void *> m_items; // Set for O(1) item existence check

  // Note: Combined queue+set structure optimizes both the common operations:
  // 1. Checking if an item exists (using the set) - O(1)
  // 2. Finding the oldest item (using the queue) - O(1)
  // This dual structure is particularly efficient for scientific simulations
  // where both operations occur frequently during cache management.
};
