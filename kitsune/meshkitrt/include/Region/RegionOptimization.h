/**
 * @file RegionOptimization.h
 * @brief Defines interfaces and classes for Region optimization strategies
 * 
 * This file defines the interfaces and classes used for Region optimization
 * strategies. These strategies determine when to optimize storage and which
 * storage mode to use based on region characteristics and usage patterns.
 */

#ifndef REGION_OPTIMIZATION_H
#define REGION_OPTIMIZATION_H

#include <functional>

namespace mesh {

// Forward declaration to avoid circular dependencies
class Region;
enum class RegionStorageMode;

/**
 * @brief Interface for Region optimization strategies
 * 
 * This interface defines the methods that must be implemented by
 * any strategy used to optimize Region storage.
 */
class RegionOptimizationStrategy {
public:
    virtual ~RegionOptimizationStrategy() = default;
    
    /**
     * @brief Decide if optimization should be performed
     * 
     * @param region Region to potentially optimize
     * @param operationCount Number of operations since last optimization
     * @return true if optimization should be performed
     */
    virtual bool shouldOptimize(
        const Region& region, size_t operationCount) const = 0;
    
    /**
     * @brief Select the optimal storage mode
     * 
     * @param region Region to optimize
     * @param currentMode Current storage mode
     * @param threshold Density threshold (0-1) for mode selection
     * @return Optimal storage mode
     */
    virtual RegionStorageMode selectOptimalMode(
        const Region& region, RegionStorageMode currentMode, double threshold) const = 0;
};

/**
 * @brief Default optimization strategy
 * 
 * This strategy optimizes after a moderate number of operations and
 * uses a density threshold of ~10% with hysteresis to prevent oscillation.
 */
class DefaultOptimizationStrategy : public RegionOptimizationStrategy {
public:
    /**
     * @brief Decide if optimization should be performed
     * 
     * The default strategy optimizes:
     * - After 100+ operations if near the threshold
     * - After 1000+ operations unconditionally
     * 
     * @param region Region to potentially optimize
     * @param operationCount Number of operations since last optimization
     * @return true if optimization should be performed
     */
    bool shouldOptimize(
        const Region& region, size_t operationCount) const override;
    
    /**
     * @brief Select the optimal storage mode
     * 
     * Uses a threshold of ~10% with hysteresis:
     * - Switch BIT_ARRAY -> CELL_SET if density < 0.07 (threshold * 0.7)
     * - Switch CELL_SET -> BIT_ARRAY if density > 0.13 (threshold * 1.3)
     * 
     * @param region Region to optimize
     * @param currentMode Current storage mode
     * @param threshold Density threshold (0-1) for mode selection
     * @return Optimal storage mode
     */
    RegionStorageMode selectOptimalMode(
        const Region& region, RegionStorageMode currentMode, double threshold) const override;
};

/**
 * @brief Optimization strategy tuned for simulation workloads
 * 
 * This strategy optimizes less frequently and prefers BIT_ARRAY mode
 * for improved set operation performance. This is beneficial for
 * simulation workloads where regions are frequently combined.
 */
class SimulationOptimizationStrategy : public RegionOptimizationStrategy {
public:
    /**
     * @brief Decide if optimization should be performed
     * 
     * The simulation strategy is more conservative:
     * - Optimize after 5000+ operations
     * - For very large regions, wait until 10000+ operations
     * 
     * @param region Region to potentially optimize
     * @param operationCount Number of operations since last optimization
     * @return true if optimization should be performed
     */
    bool shouldOptimize(
        const Region& region, size_t operationCount) const override;
    
    /**
     * @brief Select the optimal storage mode
     * 
     * Uses a lower threshold of ~5% with strong hysteresis:
     * - Switch BIT_ARRAY -> CELL_SET if density < 0.02 (threshold * 0.4)
     * - Switch CELL_SET -> BIT_ARRAY if density > 0.10 (threshold * 2.0)
     * 
     * @param region Region to optimize
     * @param currentMode Current storage mode
     * @param threshold Density threshold (0-1) for mode selection
     * @return Optimal storage mode
     */
    RegionStorageMode selectOptimalMode(
        const Region& region, RegionStorageMode currentMode, double threshold) const override;
};

} // namespace 
  
#endif // REGION_OPTIMIZATION_H

