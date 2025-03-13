/**
 * @file RegionConcepts.h
 * @brief Concepts for region operations in mesh-based simulations
 * 
 * This file defines concepts that formalize the requirements for
 * various types and callables used in the region system.
 */

#ifndef REGION_CONCEPTS_H
#define REGION_CONCEPTS_H

#include <concepts>
#include <type_traits>
#include "Cell.h" // Updated to use mesh::Cell

// Forward declarations
class Region;

// Move to mesh namespace for consistency
namespace mesh {

/**
 * @brief Concept for types that can be used as cell predicates
 * 
 * A cell predicate is a callable that takes a cell pointer and returns
 * a boolean indicating whether the cell is part of a region.
 * 
 * @tparam P The predicate type
 * @tparam C The cell type
 */
template <typename P, typename C>
concept CellPredicateConcept = requires(P pred, const C* cell) {
    { pred(cell) } -> std::convertible_to<bool>;
};

/**
 * @brief Concept for cell types in mesh-based simulations
 * 
 * Defines the minimal interface required for a cell to be used
 * with the region system.
 * 
 * @tparam C The cell type
 */
template <typename C>
concept CellConcept = requires(C cell) {
    { cell.i() } -> std::convertible_to<int>;
    { cell.j() } -> std::convertible_to<int>;
    { cell.linearIndex() } -> std::convertible_to<int>;
    { cell.mesh() } -> std::convertible_to<Mesh*>;
    
    // Ensure it inherits from Cell
    requires std::is_base_of_v<Cell, C>;
};

/**
 * @brief Concept for mesh types in mesh-based simulations
 * 
 * Defines the minimal interface required for a mesh to be used
 * with the region system.
 * 
 * @tparam M The mesh type
 * @tparam C The cell type used by the mesh
 */
template <typename M, typename C>
concept MeshConcept = requires(M mesh, int i, int j) {
    { mesh.nx() } -> std::convertible_to<int>;
    { mesh.ny() } -> std::convertible_to<int>;
    { mesh.getCell(i, j) } -> std::convertible_to<Cell*>;
    { mesh.getTypedCell(i, j) } -> std::convertible_to<C*>;
    
    // Ensure the cell type meets our Cell concept
    requires CellConcept<C>;
};

/**
 * @brief Concept for types that define region operations
 * 
 * These are callables that transform one or more regions into a new region
 * through operations like union, intersection, or filtering.
 * 
 * @tparam Op The operation type
 */
template <typename Op>
concept RegionOperationConcept = requires(Op op, const Region& r1, const Region& r2) {
    { op(r1, r2) } -> std::convertible_to<Region>;
};

/**
 * @brief Concept for region traversal strategies
 * 
 * A region traversal strategy determines the order in which cells
 * in a region are visited during iteration.
 * 
 * @tparam T The traversal strategy type
 * @tparam C The cell type
 */
template <typename T, typename C>
concept RegionTraversalStrategyConcept = requires(T strategy, const Region& region, std::vector<C*>& cells) {
    { strategy.sortCells(cells) } -> std::same_as<void>;
    { strategy.getNextCell(region, int{}) } -> std::convertible_to<C*>;
};

/**
 * @brief Concept for region cell iterators
 * 
 * Defines the requirements for iterators over cells in a region.
 * 
 * @tparam I The iterator type
 * @tparam C The cell type
 */
template <typename I, typename C>
concept RegionIteratorConcept = requires(I it, I other) {
    { *it } -> std::convertible_to<C*>;
    { ++it } -> std::convertible_to<I&>;
    { it == other } -> std::convertible_to<bool>;
    { it != other } -> std::convertible_to<bool>;
    
    // Must be at least a forward iterator
    requires std::forward_iterator<I>;
};

} // namespace mesh

#endif // REGION_CONCEPTS_H

