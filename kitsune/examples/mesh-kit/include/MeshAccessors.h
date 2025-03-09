/**
 * @file MeshAccessors.h
 * @brief Interface for efficient domain-specific operations on mesh data
 * 
 * This header includes all mesh accessor interfaces, providing a unified
 * entry point for using the various accessor types.
 */

#ifndef MESH_ACCESSORS_H
#define MESH_ACCESSORS_H

#include "MeshAccessor.h"
#include "RegionAccessor.h"
#include "FieldAccessor.h"
#include "CompoundAccessor.h"

/**
 * @namespace MeshAccessors
 * @brief Namespace containing utility functions for mesh accessors
 */
namespace MeshAccessors {
    /**
     * @brief Create a basic mesh accessor for whole-mesh operations
     * 
     * @tparam MeshType Type of the mesh
     * @tparam CellType Type of the cells
     * @param mesh Reference to the mesh
     * @return MeshAccessor<MeshType, CellType> Accessor for whole-mesh operations
     */
    template <typename MeshType, typename CellType>
    inline MeshAccessor<MeshType, CellType> createMeshAccessor(MeshType& mesh) {
        return MeshAccessor<MeshType, CellType>(mesh);
    }
    
    /**
     * @brief Create a region accessor for region-based operations
     * 
     * @tparam MeshType Type of the mesh
     * @tparam CellType Type of the cells
     * @param mesh Reference to the mesh
     * @return RegionAccessor<MeshType, CellType> Accessor for region-based operations
     */
    template <typename MeshType, typename CellType>
    inline RegionAccessor<MeshType, CellType> createRegionAccessor(MeshType& mesh) {
        return RegionAccessor<MeshType, CellType>(mesh);
    }
    
    /**
     * @brief Create a field accessor for field operations
     * 
     * @tparam MeshType Type of the mesh
     * @param mesh Reference to the mesh
     * @return FieldAccessor<MeshType> Accessor for field operations
     */
    template <typename MeshType>
    inline FieldAccessor<MeshType> createFieldAccessor(MeshType& mesh) {
        return FieldAccessor<MeshType>(mesh);
    }
    
    /**
     * @brief Create a compound accessor for combined operations
     * 
     * @tparam MeshType Type of the mesh
     * @tparam CellType Type of the cells
     * @param mesh Reference to the mesh
     * @return CompoundAccessor<MeshType, CellType> Accessor for combined operations
     */
    template <typename MeshType, typename CellType>
    inline CompoundAccessor<MeshType, CellType> createCompoundAccessor(MeshType& mesh) {
        return CompoundAccessor<MeshType, CellType>(mesh);
    }
    
    /**
     * @brief Helper for creating a union of field masks
     * 
     * @param masks Vector of field masks to union
     * @return FieldMask Union of all masks
     */
    inline FieldMask unionFieldMasks(const std::vector<FieldMask>& masks) {
        FieldMask result = 0;
        for (const auto& mask : masks) {
            result |= mask;
        }
        return result;
    }
    
    /**
     * @brief Helper for creating a union of region masks
     * 
     * @param masks Vector of region masks to union
     * @return RegionMask Union of all masks
     */
    inline RegionMask unionRegionMasks(const std::vector<RegionMask>& masks) {
        RegionMask result = 0;
        for (const auto& mask : masks) {
            result |= mask;
        }
        return result;
    }
}

#endif // MESH_ACCESSORS_H


