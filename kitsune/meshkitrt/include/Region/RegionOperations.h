/**
 * @file RegionOperations.h
 * @brief Standalone functions for region operations
 * 
 * This file provides a set of free functions for creating and manipulating
 * regions, designed to be usable without a central manager class.
 */

#ifndef REGION_OPERATIONS_H
#define REGION_OPERATIONS_H

#include "Region.h"
#include "RegionConcepts.h"
#include "RegionUtils.h"
#include <functional>
#include <vector>
#include <memory>
#include <string>
#include <random>

namespace mesh {

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
template <typename MeshT, typename CellT, CellPredicateConcept<CellT> PredT>
Region filterMesh(
    const MeshT& mesh,
    PredT predicate,
    const std::string& name = "FilteredRegion"
) {
    // Create a type-erased wrapper for the predicate
    std::function<bool(const Cell*)> basePredicate = 
        [predicate](const Cell* cell) {
            return predicate(static_cast<const CellT*>(cell));
        };
    
    // Create a region definition
    auto definition = std::make_shared<PredicateRegion>(name, basePredicate);
    
    // Create a stable ID based on name and current time
    std::hash<std::string> hasher;
    RegionID id = static_cast<RegionID>(hasher(name) % 1000000);
    
    // Create region with mesh binding
    Region result(id, definition, const_cast<MeshT*>(&mesh));
    
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

    // Create region with mesh binding
    Region result(id, definition, const_cast<MeshT*>(&mesh));

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
template <typename MeshT, typename CellT, CellPredicateConcept<CellT> PredT>
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
 * Creates a new region representing all cells that are in either input region.
 * The resulting region will have a storage mode optimized for the characteristics
 * of the union operation.
 * 
 * @param regionA First region
 * @param regionB Second region
 * @param name Optional name for the region
 * @return Region representing the union
 */
inline Region createUnionRegion(
    const Region& regionA,
    const Region& regionB,
    const std::string& name = "UnionRegion") {
    // Validate that both regions use the same mesh
    if (regionA.mesh() != regionB.mesh()) {
        throw std::invalid_argument("Cannot create union of regions bound to different meshes");
    }
    
    // Create a composite region definition for the union
    auto compDef = std::make_shared<CompositeRegion>(
          name,
          regionA.getDefinition(),
          regionB.getDefinition(),
          CompositeRegion::Operation::UNION
    );

    // Create a stable ID using the robust composite ID generator
    // Use operation code 1 for union
    RegionID id = generateCompositeID(regionA.getId(), regionB.getId(), 1);
    
    // Create new region with the composite definition and the same mesh
    Region result(id, compDef, regionA.mesh());
    
    // Optimal storage selection based on storage modes of inputs
    // If both regions use the same storage mode, use that mode
    if (regionA.getStorageMode() == regionB.getStorageMode()) {
        result.setStorageMode(regionA.getStorageMode());
        
        if (regionA.getStorageMode() == RegionStorageMode::CELL_SET) {
            // For cell sets, perform set union
            auto& resultIndices = result.getCellIndicesForWrite();
            const auto& indicesA = regionA.getCellIndices();
            const auto& indicesB = regionB.getCellIndices();
            
            // Reserve space for efficiency
            resultIndices.reserve(indicesA.size() + indicesB.size());
            
            // Insert all indices from both regions
            resultIndices.insert(indicesA.begin(), indicesA.end());
            resultIndices.insert(indicesB.begin(), indicesB.end());
        }
        else if (regionA.getStorageMode() == RegionStorageMode::BIT_ARRAY) {
            // For bit arrays, perform bitwise OR
            BitArray resultBits = regionA.getBitArray();
            resultBits.bitwiseOr(regionB.getBitArray());
            result.setBitArray(std::move(resultBits));
        }
    }
    else {
        // Different storage modes - choose based on estimated density
        double densityA = static_cast<double>(regionA.size()) / regionA.getMeshSize();
        double densityB = static_cast<double>(regionB.size()) / regionB.getMeshSize();
        
        // Estimate density of union: P(A∪B) = P(A) + P(B) - P(A∩B)
        // Assuming independence: P(A∩B) ≈ P(A)P(B)
        double estimatedDensity = densityA + densityB - (densityA * densityB);
        
        if (estimatedDensity > 0.1) {
            // Dense result - use bit array
            BitArray resultBits = regionA.getBitArray();
            resultBits.bitwiseOr(regionB.getBitArray());
            result.setBitArray(std::move(resultBits));
        }
        else {
            // Sparse result - use cell set
            auto& resultIndices = result.getCellIndicesForWrite();
            const auto& indicesA = regionA.getCellIndices();
            const auto& indicesB = regionB.getCellIndices();
            
            // Reserve space for efficiency
            resultIndices.reserve(indicesA.size() + indicesB.size());
            
            // Insert all indices from both regions
            resultIndices.insert(indicesA.begin(), indicesA.end());
            resultIndices.insert(indicesB.begin(), indicesB.end());
        }
    }
    
    return result;
}

/**
 * @brief Create an intersection of two regions
 * 
 * Creates a new region representing cells that are in both input regions.
 * The resulting region will have a storage mode optimized for the characteristics
 * of the intersection operation.
 * 
 * @param regionA First region
 * @param regionB Second region
 * @param name Optional name for the region
 * @return Region representing the intersection
 */
inline Region createIntersectionRegion(
    const Region& regionA,
    const Region& regionB,
    const std::string& name = "IntersectionRegion") {
    // Validate that both regions use the same mesh
    if (regionA.mesh() != regionB.mesh()) {
        throw std::invalid_argument("Cannot create intersection of regions bound to different meshes");
    }
    
    // Create a composite region definition for the intersection
    auto compDef = std::make_shared<CompositeRegion>(
        name,
        regionA.getDefinition(),
        regionB.getDefinition(),
        CompositeRegion::Operation::INTERSECTION
    );
    
    // Create a stable ID using the robust composite ID generator
    // Use operation code 2 for intersection
    RegionID id = generateCompositeID(regionA.getId(), regionB.getId(), 2);
    
    // Create new region with the composite definition and the same mesh
    Region result(id, compDef, regionA.mesh());
    
    // For intersection, it's usually most efficient to use the same storage mode
    // as the smaller of the two regions, since the result can't be larger than that
    
    // Get sizes to determine which region is smaller
    size_t sizeA = regionA.size();
    size_t sizeB = regionB.size();
    
    if (sizeA < sizeB) {
        // Region A is smaller, process its elements
        if (regionA.getStorageMode() == RegionStorageMode::CELL_SET) {
            // Use cell set for the result
            result.setStorageMode(RegionStorageMode::CELL_SET);
            auto& resultIndices = result.getCellIndicesForWrite();
            const auto& indicesA = regionA.getCellIndices();
            
            // Reserve space for efficiency (worst case: all of smaller region)
            resultIndices.reserve(indicesA.size());
            
            // Add indices that are in both regions
            for (int idx : indicesA) {
                if (regionB.containsIndex(idx)) {
                    resultIndices.insert(idx);
                }
            }
        }
        else {
            // Use bit array for the result
            BitArray resultBits = regionA.getBitArray();
            resultBits.bitwiseAnd(regionB.getBitArray());
            result.setBitArray(std::move(resultBits));
        }
    }
    else {
        // Region B is smaller (or they're equal), process its elements
        if (regionB.getStorageMode() == RegionStorageMode::CELL_SET) {
            // Use cell set for the result
            result.setStorageMode(RegionStorageMode::CELL_SET);
            auto& resultIndices = result.getCellIndicesForWrite();
            const auto& indicesB = regionB.getCellIndices();
            
            // Reserve space for efficiency (worst case: all of smaller region)
            resultIndices.reserve(indicesB.size());
            
            // Add indices that are in both regions
            for (int idx : indicesB) {
                if (regionA.containsIndex(idx)) {
                    resultIndices.insert(idx);
                }
            }
        }
        else {
            // Use bit array for the result
            BitArray resultBits = regionB.getBitArray();
            resultBits.bitwiseAnd(regionA.getBitArray());
            result.setBitArray(std::move(resultBits));
        }
    }
    
    // Automatically optimize storage based on result density
    result.optimizeStorage();
    
    return result;
}

/**
 * @brief Create a difference of two regions (A - B)
 * 
 * Creates a new region representing cells that are in the first region but not
 * in the second region. The resulting region will have a storage mode optimized 
 * for the characteristics of the difference operation.
 * 
 * @param regionA First region (minuend)
 * @param regionB Second region (subtrahend)
 * @param name Optional name for the region
 * @return Region representing the difference
 */
inline Region createDifferenceRegion(
    const Region& regionA,
    const Region& regionB,
    const std::string& name = "DifferenceRegion"
) {
    // Validate that both regions use the same mesh
    if (regionA.mesh() != regionB.mesh()) {
        throw std::invalid_argument("Cannot create difference of regions bound to different meshes");
    }
    
    // Create a composite region definition for the difference
    auto compDef = std::make_shared<CompositeRegion>(
        name,
        regionA.getDefinition(),
        regionB.getDefinition(),
        CompositeRegion::Operation::DIFFERENCE
    );
    
    // Create a stable ID using the robust composite ID generator
    // Use operation code 3 for difference
    RegionID id = generateCompositeID(regionA.getId(), regionB.getId(), 3);
    
    // Create new region with the composite definition and the same mesh
    Region result(id, compDef, regionA.mesh());
    
    // For difference, the result can't be larger than region A, so we
    // can use the same storage mode
    if (regionA.getStorageMode() == RegionStorageMode::CELL_SET) {
        // Use cell set for the result
        result.setStorageMode(RegionStorageMode::CELL_SET);
        auto& resultIndices = result.getCellIndicesForWrite();
        const auto& indicesA = regionA.getCellIndices();
        
        // Reserve space for efficiency (worst case: all of first region)
        resultIndices.reserve(indicesA.size());
        
        // Add indices that are in A but not in B
        for (int idx : indicesA) {
            if (!regionB.containsIndex(idx)) {
                resultIndices.insert(idx);
            }
        }
    }
    else {
        // Use bit array for the result
        BitArray resultBits = regionA.getBitArray();
        resultBits.bitwiseAndNot(regionB.getBitArray());
        result.setBitArray(std::move(resultBits));
    }
    
    // Automatically optimize storage based on result density
    result.optimizeStorage();
    
    return result;
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
 */
template <typename MeshT, typename CellT, typename FuncT>
requires std::invocable<FuncT, CellT*>
void forEachCellInRegion(
    MeshT& mesh,
    const Region& region,
    FuncT func
) {
    // Choose the most efficient iteration strategy based on storage mode
    if (region.getStorageMode() == RegionStorageMode::CELL_SET) {
        // Iterate through explicit cell indices
        const auto& indices = region.getCellIndices();
        for (int idx : indices) {
            // Convert linear index to 2D coordinates
            int i = idx % mesh.nx();
            int j = idx / mesh.nx();
            
            // Get the cell and call the function
            CellT* cell = mesh.getTypedCell(i, j);
            if (cell) {
                func(cell);
            }
        }
    }
    else if (region.getStorageMode() == RegionStorageMode::BIT_ARRAY) {
        // Iterate through bit array using efficient findFirst/findNext
        const BitArray& bitArray = region.getBitArray();
        
        for (size_t idx = bitArray.findFirst(); idx < bitArray.size(); idx = bitArray.findNext(idx)) {
            // Convert linear index to 2D coordinates
            int i = static_cast<int>(idx) % mesh.nx();
            int j = static_cast<int>(idx) / mesh.nx();
            
            // Get the cell and call the function
            CellT* cell = mesh.getTypedCell(i, j);
            if (cell) {
                func(cell);
            }
        }
    }
    else if (region.getStorageMode() == RegionStorageMode::DYNAMIC) {
        // For dynamic regions, iterate through all cells in the mesh
        for (int j = 0; j < mesh.ny(); ++j) {
            for (int i = 0; i < mesh.nx(); ++i) {
                CellT* cell = mesh.getTypedCell(i, j);
                if (cell && region.contains(cell)) {
                    func(cell);
                }
            }
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
 * @return Vector of pointers to cells in the region
 */
template <typename MeshT, typename CellT>
std::vector<CellT*> getCellsInRegion(
    MeshT& mesh,
    const Region& region
) {
    std::vector<CellT*> cells;
    
    // Reserve space based on region size for efficiency
    cells.reserve(region.size());
    
    // Use forEachCellInRegion to fill the vector
    forEachCellInRegion<MeshT, CellT>(mesh, region, [&cells](CellT* cell) {
        cells.push_back(cell);
    });
    
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
    const Cell* cell,
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

/**
 * @brief Create a region containing cells that match a property value
 * 
 * @tparam MeshT Type of mesh
 * @tparam CellT Type of cell
 * @tparam PropertyT Type of property
 * @param mesh Mesh to create region from
 * @param propertyAccessor Function that gets property value from a cell
 * @param value Property value to match
 * @param name Optional name for the region
 * @return Region object
 */
template <typename MeshT, typename CellT, typename PropertyT>
Region createRegionByProperty(
    const MeshT& mesh,
    std::function<PropertyT(const CellT*)> propertyAccessor,
    const PropertyT& value,
    const std::string& name = "PropertyRegion"
) {
    return filterMesh<MeshT, CellT>(
        mesh,
        [propertyAccessor, value](const CellT* cell) {
            return propertyAccessor(cell) == value;
        },
        name
    );
}

/**
 * @brief Create a region containing cells that satisfy a property condition
 * 
 * @tparam MeshT Type of mesh
 * @tparam CellT Type of cell
 * @tparam PropertyT Type of property
 * @param mesh Mesh to create region from
 * @param propertyAccessor Function that gets property value from a cell
 * @param condition Function that evaluates property value
 * @param name Optional name for the region
 * @return Region object
 */
template <typename MeshT, typename CellT, typename PropertyT>
Region createRegionByPropertyCondition(
    const MeshT& mesh,
    std::function<PropertyT(const CellT*)> propertyAccessor,
    std::function<bool(const PropertyT&)> condition,
    const std::string& name = "PropertyConditionRegion"
) {
    return filterMesh<MeshT, CellT>(
        mesh,
        [propertyAccessor, condition](const CellT* cell) {
            return condition(propertyAccessor(cell));
        },
        name
    );
}

} // namespace mesh

#endif // REGION_OPERATIONS_H


