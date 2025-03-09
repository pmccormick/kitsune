/**
 * @file a
 * @brief Standalone functions for region operations
 * 
 * This file provides a set of free functions for creating and manipulating
 * regions, designed to be usable without a central manager class.
 */

#ifndef REGION_OPERATIONS_H
#define REGION_OPERATIONS_H

#include "Region.h"
#include "RegionConcepts.h"
#include <functional>
#include <vector>
#include <memory>
#include <string>
#include <random>

/**
 * @brief Create a region containing cells that match a predicate
 * 
 * @tparam MeshT Type of mesh
 * @tparam CellT Type of cell
 * @tparam PredT Type of predicate function
 * @param mesh Mesh to create region from
 * @param predicate Function determining if a cell is in the region
 * @param name Optional name for the region
 * @return Region object
 */
template <typename MeshT, typename CellT, CellPredicate<CellT> PredT>
Region filterMesh(
    const MeshT& mesh,
    PredT predicate,
    const std::string& name = "FilteredRegion"
) {
    // Create a type-erased wrapper for the predicate
    std::function<bool(const CellBase*)> basePredicate = 
        [predicate](const CellBase* cell) {
            return predicate(static_cast<const CellT*>(cell));
        };
    
    // Create a region definition
    auto definition = std::make_shared<PredicateRegion>(name, basePredicate);
    
    // Create the region with a stable ID based on name and current time
    std::hash<std::string> hasher;
    RegionID id = static_cast<RegionID>(hasher(name) % 1000000);
    
    // Create and build the region
    Region result(id, definition, mesh.nx() * mesh.ny());
    result.buildFromMesh(mesh);
    
    return result;
}

/**
 * @brief Create a rectangular region
 * 
 * @tparam MeshT Type of mesh
 * @param mesh Mesh to create region from
 * @param minI Minimum i-index (inclusive)
 * @param minJ Minimum j-index (inclusive) 
 * @param maxI Maximum i-index (inclusive)
 * @param maxJ Maximum j-index (inclusive)
 * @param name Optional name for the region
 * @return Region object
 */
template <typename MeshT>
Region createRectangularRegion(
    const MeshT& mesh,
    int minI, int minJ, int maxI, int maxJ,
    const std::string& name = "RectangularRegion"
) {
    // Create a rectangular region definition
    auto definition = std::make_shared<RectangularRegion>(name, minI, minJ, maxI, maxJ);
    
    // Create a stable ID based on coordinates and name
    std::hash<std::string> hasher;
    std::string idStr = name + ":" + 
                        std::to_string(minI) + "," + 
                        std::to_string(minJ) + "," + 
                        std::to_string(maxI) + "," + 
                        std::to_string(maxJ);
    RegionID id = static_cast<RegionID>(hasher(idStr) % 1000000);
    
    // Create and build the region
    Region result(id, definition, mesh.nx() * mesh.ny());
    result.buildFromMesh(mesh);
    
    return result;
}

/**
 * @brief Create a filtered region from an existing region
 * 
 * @tparam MeshT Type of mesh
 * @tparam CellT Type of cell
 * @tparam PredT Type of predicate function
 * @param mesh Mesh containing the cells
 * @param sourceRegion Source region to filter
 * @param predicate Function determining if a cell should be included
 * @param name Optional name for the result
 * @return Region object containing the filtered subset of cells
 */
template <typename MeshT, typename CellT, CellPredicate<CellT> PredT>
Region filterRegion(
    const MeshT& mesh,
    const Region& sourceRegion,
    PredT predicate,
    const std::string& name = "FilteredRegion"
) {
    // Create a combined predicate that checks both region membership and the custom predicate
    auto combinedPredicate = [&sourceRegion, predicate](const CellT* cell) {
        return sourceRegion.contains(cell) && predicate(cell);
    };
    
    // Use the filterMesh function with our combined predicate
    return filterMesh<MeshT, CellT>(mesh, combinedPredicate, name);
}

/**
 * @brief Create a union of two regions
 * 
 * @param regionA First region
 * @param regionB Second region
 * @param name Optional name for the region
 * @return Region representing the union
 */
inline Region createUnionRegion(
    const Region& regionA,
    const Region& regionB,
    const std::string& name = "UnionRegion"
) {
    // Use the Region class method to create the union
    return regionA.createUnion(regionB, name, regionA.getId() + regionB.getId());
}

/**
 * @brief Create an intersection of two regions
 * 
 * @param regionA First region
 * @param regionB Second region
 * @param name Optional name for the region
 * @return Region representing the intersection
 */
inline Region createIntersectionRegion(
    const Region& regionA,
    const Region& regionB,
    const std::string& name = "IntersectionRegion"
) {
    // Use the Region class method to create the intersection
    return regionA.createIntersection(regionB, name, regionA.getId() * regionB.getId());
}

/**
 * @brief Create a difference of two regions
 * 
 * @param regionA First region
 * @param regionB Second region
 * @param name Optional name for the region
 * @return Region representing the difference
 */
inline Region createDifferenceRegion(
    const Region& regionA,
    const Region& regionB,
    const std::string& name = "DifferenceRegion"
) {
    // Use the Region class method to create the difference
    return regionA.createDifference(regionB, name, regionA.getId() - regionB.getId());
}

/**
 * @brief Apply a function to each cell in a region
 * 
 * @tparam MeshT Type of mesh
 * @tparam CellT Type of cell
 * @tparam FuncT Type of function to apply
 * @param mesh Mesh containing the cells
 * @param region Region to iterate over
 * @param func Function to apply to each cell
 * @param order Traversal order for cells
 */
template <typename MeshT, typename CellT, typename FuncT>
requires std::invocable<FuncT, CellT*>
void forEachCellInRegion(
    MeshT& mesh,
    const Region& region,
    FuncT func,
    RegionTraversalOrder order = RegionTraversalOrder::NATURAL
) {
    // Use the RegionCellRange and iterator from the previous implementation
    for (CellT* cell : RegionCellRange<MeshT, CellT>(mesh, region, order)) {
        if (cell) {
            func(cell);
        }
    }
}

/**
 * @brief Get all cells in a region
 * 
 * @tparam MeshT Type of mesh
 * @tparam CellT Type of cell
 * @param mesh Mesh containing the cells
 * @param region Region to get cells from
 * @param order Traversal order for cells
 * @return Vector of pointers to cells in the region
 */
template <typename MeshT, typename CellT>
std::vector<CellT*> getCellsInRegion(
    MeshT& mesh,
    const Region& region,
    RegionTraversalOrder order = RegionTraversalOrder::NATURAL
) {
    std::vector<CellT*> cells;
    
    forEachCellInRegion<MeshT, CellT>(mesh, region, [&cells](CellT* cell) {
        cells.push_back(cell);
    }, order);
    
    return cells;
}

/**
 * @brief Check if a cell is in a region
 * 
 * @param cell Cell to check
 * @param region Region to check
 * @return true if the cell is in the region
 */
inline bool isCellInRegion(
    const CellBase* cell,
    const Region& region
) {
    return region.contains(cell);
}

/**
 * @brief Create a region containing boundary cells of a mesh
 * 
 * @tparam MeshT Type of mesh
 * @tparam CellT Type of cell
 * @param mesh Mesh to create region from
 * @param name Optional name for the region
 * @return Region object containing boundary cells
 */
template <typename MeshT, typename CellT>
Region createBoundaryRegion(
    const MeshT& mesh,
    const std::string& name = "BoundaryRegion"
) {
    return filterMesh<MeshT, CellT>(
        mesh,
        [&mesh](const CellT* cell) {
            return cell->i() == 0 || cell->j() == 0 || 
                   cell->i() == mesh.nx() - 1 || cell->j() == mesh.ny() - 1;
        },
        name
    );
}

/**
 * @brief Create a region containing interior cells of a mesh
 * 
 * @tparam MeshT Type of mesh
 * @tparam CellT Type of cell
 * @param mesh Mesh to create region from
 * @param name Optional name for the region
 * @return Region object containing interior cells
 */
template <typename MeshT, typename CellT>
Region createInteriorRegion(
    const MeshT& mesh,
    const std::string& name = "InteriorRegion"
) {
    return filterMesh<MeshT, CellT>(
        mesh,
        [&mesh](const CellT* cell) {
            return cell->i() > 0 && cell->j() > 0 && 
                   cell->i() < mesh.nx() - 1 && cell->j() < mesh.ny() - 1;
        },
        name
    );
}

#endif // REGION_OPERATIONS_H


