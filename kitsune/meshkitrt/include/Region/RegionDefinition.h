/**
 * @file RegionDefinition.h
 * @brief Defines the region definition classes for specifying region membership criteria
 * 
 * This file contains the abstract RegionDefinition base class and its concrete implementations
 * that define how regions are constructed (via predicates, geometric shapes, or compositions).
 */

#ifndef REGION_DEFINITION_H
#define REGION_DEFINITION_H

#include "Cell.h"
#include <climits>
#include <string>
#include <unordered_set>
#include <vector>
#include <memory>
#include <functional>
#include <stdexcept>
#include <cassert>

// Forward declarations
namespace mesh {
    class Mesh;

/**
 * @brief Region identifier type
 */
using RegionID = uint32_t;

/**
 * @brief Abstract base class for region definitions
 * 
 * RegionDefinition provides a polymorphic interface for defining
 * the membership criteria of a region (geometric, predicates, etc.)
 */
class RegionDefinition {
public:
    virtual ~RegionDefinition() = default;
    
    /**
     * @brief Check if a cell belongs to this region
     * 
     * @param cell Pointer to the cell to check
     * @return true if the cell is in the region
     */
    virtual bool contains(const Cell* cell) const = 0;
    
    /**
     * @brief Get the name of the region definition
     * 
     * @return Region definition name
     */
    virtual std::string getName() const = 0;
    
    /**
     * @brief Get a bounding box for the region (if applicable)
     * 
     * Returns a pair of coordinate pairs that bound the region in the mesh.
     * This is used to optimize region building by limiting the cells that need
     * to be checked for membership.
     * 
     * @return Pair of min/max indices that bound the region
     */
    virtual std::pair<std::pair<int, int>, std::pair<int, int>> getBounds() const {
        // Create pairs explicitly with maximum extents for safety
        std::pair<int, int> minPair(0, 0);
        std::pair<int, int> maxPair(INT_MAX, INT_MAX);
        return std::make_pair(minPair, maxPair);
    }
};

/**
 * @brief Predicate-based region definition
 * 
 * Defines a region using a function that determines membership.
 * This allows for arbitrary custom region shapes and conditions.
 */
class PredicateRegion : public RegionDefinition {
public:
    /**
     * @brief Construct a new Predicate Region
     * 
     * @param name Region name
     * @param predicate Function that determines if a cell is in the region
     */
    PredicateRegion(std::string name, std::function<bool(const Cell*)> predicate)
        : m_name(std::move(name)), m_predicate(std::move(predicate)) {}
    
    /**
     * @brief Check if a cell belongs to this region using the predicate
     * 
     * @param cell Pointer to the cell to check
     * @return true if the predicate returns true for this cell, false otherwise
     */
    bool contains(const Cell* cell) const override {
        if (!cell) return false;
        return m_predicate(cell);
    }
    
    /**
     * @brief Get the name of the region definition
     * 
     * @return Region definition name
     */
    std::string getName() const override {
        return m_name;
    }
    
private:
    std::string m_name;                             ///< Name of the region
    std::function<bool(const Cell*)> m_predicate; ///< Function determining membership
};

/**
 * @brief Rectangular region definition
 * 
 * Defines a region as a rectangular area in the mesh.
 * This is a common and efficient region shape for many applications.
 */
class RectangularRegion : public RegionDefinition {
public:
    /**
     * @brief Construct a new Rectangular Region
     * 
     * @param name Region name
     * @param minI Minimum i-index (inclusive)
     * @param minJ Minimum j-index (inclusive)
     * @param maxI Maximum i-index (inclusive)
     * @param maxJ Maximum j-index (inclusive)
     */
    RectangularRegion(std::string name, int minI, int minJ, int maxI, int maxJ)
        : m_name(std::move(name)), m_minI(minI), m_minJ(minJ), m_maxI(maxI), m_maxJ(maxJ) {}
    
    /**
     * @brief Check if a cell belongs to this region by testing if it's within bounds
     * 
     * @param cell Pointer to the cell to check
     * @return true if the cell is within the defined rectangle, false otherwise
     */
    bool contains(const Cell* cell) const override {
        if (!cell) return false;
        
        int i = cell->i();
        int j = cell->j();
        return i >= m_minI && i <= m_maxI && j >= m_minJ && j <= m_maxJ;
    }
    
    /**
     * @brief Get the name of the region definition
     * 
     * @return Region definition name
     */
    std::string getName() const override {
        return m_name;
    }
    
    /**
     * @brief Get the bounding box for the rectangular region
     * 
     * For rectangular regions, this returns the exact bounds of the rectangle.
     * 
     * @return Pair of min/max indices that exactly define the rectangle
     */
    std::pair<std::pair<int, int>, std::pair<int, int>> getBounds() const override {
        return {{m_minI, m_minJ}, {m_maxI, m_maxJ}};
    }
    
private:
    std::string m_name;  ///< Name of the region
    int m_minI, m_minJ;  ///< Lower bounds (inclusive)
    int m_maxI, m_maxJ;  ///< Upper bounds (inclusive)
};

/**
 * @brief Composite region definition using set operations
 * 
 * Combines two region definitions using set operations (union, intersection, difference).
 * This allows for building complex regions from simpler components.
 */
class CompositeRegion : public RegionDefinition {
public:
    /**
     * @brief Set operation types
     */
    enum class Operation {
        UNION,        ///< A ∪ B (cells in either region)
        INTERSECTION, ///< A ∩ B (cells in both regions)
        DIFFERENCE    ///< A - B (cells in A but not in B)
    };
    
    /**
     * @brief Construct a new Composite Region
     * 
     * @param name Region name
     * @param regionA First region definition
     * @param regionB Second region definition
     * @param operation Set operation to apply
     */
    CompositeRegion(std::string name, 
                   std::shared_ptr<const RegionDefinition> regionA,
                   std::shared_ptr<const RegionDefinition> regionB,
                   Operation operation)
        : m_name(std::move(name)),
          m_regionA(std::move(regionA)),
          m_regionB(std::move(regionB)),
          m_operation(operation) { }

    /**
     * @brief Check if a cell belongs to this region using the composite operation
     * 
     * Evaluates the set operation between the two component regions.
     * 
     * @param cell Pointer to the cell to check
     * @return true if the cell satisfies the composite criteria, false otherwise
     */
    bool contains(const Cell* cell) const override;
    
    /**
     * @brief Get the name of the region definition
     * 
     * @return Region definition name
     */
    std::string getName() const override;
    
    /**
     * @brief Get the bounding box for the composite region
     * 
     * For UNION: combines the bounds of both regions
     * For INTERSECTION/DIFFERENCE: uses the bounds of the first region
     * 
     * @return Pair of min/max indices that bound the region
     */
    std::pair<std::pair<int, int>, std::pair<int, int>> getBounds() const override;
    
    /**
     * @brief Get the operation type for this composite region
     * 
     * @return Operation type
     */
    Operation getOperation() const;
    
    /**
     * @brief Get the first region definition
     * 
     * @return Pointer to the first region definition
     */
    std::shared_ptr<const RegionDefinition> getRegionA() const {
        return m_regionA;
    }
    
    /**
     * @brief Get the second region definition
     * 
     * @return Pointer to the second region definition
     */
    std::shared_ptr<const RegionDefinition> getRegionB() const {
        return m_regionB;
    }
    
private:
    std::string m_name;                                 ///< Name of the region
    std::shared_ptr<const RegionDefinition> m_regionA;  ///< First region definition
    std::shared_ptr<const RegionDefinition> m_regionB;  ///< Second region definition
    Operation m_operation;                              ///< Set operation to apply
};

} // namespace 
 
#endif // REGION_DEFINITION_H
 
