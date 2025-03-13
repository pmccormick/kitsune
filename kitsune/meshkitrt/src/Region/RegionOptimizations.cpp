/**
 * @file RegionOptimization.cpp
 * @brief Implementation of Region optimization strategy classes
 * 
 * This file implements the various optimization strategies defined in
 * RegionOptimization.h. These strategies determine when to optimize
 * storage and which storage mode to use based on region characteristics.
 */

#include "RegionOptimization.h"
#include "Region.h"
#include <cmath>
#include <stdexcept>

namespace mesh {

bool DefaultOptimizationStrategy::shouldOptimize(
    const Region& region, size_t operationCount) const 
{
    // Don't optimize too frequently
    if (operationCount < 100) return false;
    
    // Always optimize after a significant number of operations
    if (operationCount >= 1000) return true;
    
    // Calculate density
    double density = 0.0;
    try {
        density = static_cast<double>(region.size()) / region.getMeshSize();
    } catch (const std::exception&) {
        // If we can't calculate density (e.g., in DYNAMIC mode), don't optimize
        return false;
    }
    
    // Optimize more frequently when near the decision boundary
    bool nearThreshold = std::abs(density - 0.1) < 0.02;
    
    return (nearThreshold && operationCount >= 500);
}

RegionStorageMode DefaultOptimizationStrategy::selectOptimalMode(
    const Region& region, RegionStorageMode currentMode, double threshold) const 
{
    // Calculate density
    double density = 0.0;
    try {
        density = static_cast<double>(region.size()) / region.getMeshSize();
    } catch (const std::exception&) {
        // If we can't calculate density, stay in current mode
        return currentMode;
    }
    
    // Use hysteresis to prevent oscillation
    if (currentMode == RegionStorageMode::BIT_ARRAY) {
        // Stay in BIT_ARRAY mode unless density is significantly below threshold
        return (density > threshold * 0.7) ? 
            RegionStorageMode::BIT_ARRAY : RegionStorageMode::CELL_SET;
    } else {
        // Stay in CELL_SET mode unless density is significantly above threshold
        return (density > threshold * 1.3) ? 
            RegionStorageMode::BIT_ARRAY : RegionStorageMode::CELL_SET;
    }
}

bool SimulationOptimizationStrategy::shouldOptimize(
    const Region& region, size_t operationCount) const 
{
    // For simulations, optimize less frequently
    if (operationCount < 5000) return false;
    
    // For very large regions, be even more conservative
    if (region.getMeshSize() > 1000000) {
        return operationCount > 10000;
    }
    
    return true;
}

RegionStorageMode SimulationOptimizationStrategy::selectOptimalMode(
    const Region& region, RegionStorageMode currentMode, double threshold) const 
{
    // Calculate density
    double density = 0.0;
    try {
        density = static_cast<double>(region.size()) / region.getMeshSize();
    } catch (const std::exception&) {
        // If we can't calculate density, stay in current mode
        return currentMode;
    }
    
    // For simulations, prefer BIT_ARRAY for improved set operation performance
    // Use a lower threshold (0.05 instead of 0.1)
    double simThreshold = threshold > 0 ? threshold : 0.05; // Default to 0.05 if no threshold specified
    
    // With stronger hysteresis to avoid mode switching during simulation
    if (currentMode == RegionStorageMode::BIT_ARRAY) {
        // Stay in BIT_ARRAY mode unless density is very low
        return (density > simThreshold * 0.4) ? 
            RegionStorageMode::BIT_ARRAY : RegionStorageMode::CELL_SET;
    } else {
        // Only switch to BIT_ARRAY if density is definitively high
        return (density > simThreshold * 2.0) ? 
            RegionStorageMode::BIT_ARRAY : RegionStorageMode::CELL_SET;
    }
}

} // namespace mesh 
