/**
 * @file AccessorExtensions.h
 * @brief Iterator extensions for accessor classes
 * 
 * This file contains extension methods to be added to the various accessor classes
 * to integrate iterator support. These extensions provide a consistent interface
 * for traversing mesh elements using modern C++ iteration patterns.
 */

#ifndef ACCESSOR_EXTENSIONS_H
#define ACCESSOR_EXTENSIONS_H

#include "AccessorIterators.h"

/**
 * Extensions for MeshAccessor class
 */
namespace MeshAccessorExtension {

template <typename MeshType, typename CellType>
class Extension {
private:
    // Iterator implementation
    using Iterators = MeshAccessorIterators<MeshType, CellType>;
    
public:
    // Iterator type aliases
    using CellIterator = typename Iterators::CellIterator;
    using InteriorCellIterator = typename Iterators::InteriorCellIterator;
    using BoundaryCellIterator = typename Iterators::BoundaryCellIterator;
    using BlockCellIterator = typename Iterators::BlockCellIterator;
    
    // Range type aliases
    using CellRange = typename Iterators::CellRange;
    using InteriorCellRange = typename Iterators::InteriorCellRange;
    using BoundaryCellRange = typename Iterators::BoundaryCellRange;
    using BlockCellRange = typename Iterators::BlockCellRange;
    
    // Basic cell iteration
    CellIterator begin(MeshType& mesh) { return CellIterator(mesh); }
    CellIterator end(MeshType& mesh) { return CellIterator(mesh, 0, mesh.ny()); }
    
    // Range-based iteration
    CellRange cells(MeshType& mesh) { return CellRange(mesh); }
    InteriorCellRange interiorCells(MeshType& mesh) { return InteriorCellRange(mesh); }
    BoundaryCellRange boundaryCells(MeshType& mesh) { return BoundaryCellRange(mesh); }
    BlockCellRange blockCells(MeshType& mesh, int blockSizeX = 16, int blockSizeY = 16) {
        return BlockCellRange(mesh, blockSizeX, blockSizeY);
    }
};

} // namespace MeshAccessorExtension

/**
 * Extensions for RegionAccessor class
 */
namespace RegionAccessorExtension {

template <typename MeshType, typename CellType>
class Extension {
private:
    // Iterator implementation
    using Iterators = RegionAccessorIterators<MeshType, CellType>;
    
public:
    // Iterator type aliases
    using RegionCellIterator = typename Iterators::RegionCellIterator;
    
    // Range type aliases
    using RegionCellRange = typename Iterators::RegionCellRange;
    
    // Range-based iteration
    RegionCellRange cells(MeshType& mesh, RegionMask regionMask) {
        return RegionCellRange(mesh, regionMask);
    }
    
    RegionCellRange cells(MeshType& mesh, RegionID regionID) {
        RegionMask mask = mesh.getRegionMask(regionID);
        return RegionCellRange(mesh, mask);
    }
};

} // namespace RegionAccessorExtension

/**
 * Extensions for FieldAccessor class
 */
namespace FieldAccessorExtension {

template <typename MeshType>
class Extension {
private:
    // Iterator implementation
    using Iterators = FieldAccessorIterators<MeshType>;
    
public:
    // Iterator type template
    template <typename T, typename LocationTag>
    using FieldIterator = typename Iterators::template FieldIterator<T, LocationTag>;
    
    // Range type template
    template <typename T, typename LocationTag>
    using FieldRange = typename Iterators::template FieldRange<T, LocationTag>;
    
    // Range-based iteration
    template <typename T, typename LocationTag>
    FieldRange<T, LocationTag> fields(MeshType& mesh, FieldMask fieldMask) {
        return FieldRange<T, LocationTag>(mesh, fieldMask);
    }
    
    template <typename T, typename LocationTag>
    FieldRange<T, LocationTag> fields(MeshType& mesh, const std::vector<FieldID>& fieldIDs, 
                                     FieldAccessor<MeshType>& accessor) {
        FieldMask mask = accessor.createMask(fieldIDs);
        return FieldRange<T, LocationTag>(mesh, mask);
    }
};

} // namespace FieldAccessorExtension

/**
 * Extensions for CompoundAccessor class
 */
namespace CompoundAccessorExtension {

template <typename MeshType, typename CellType>
class Extension {
private:
    // Iterator implementation
    using Iterators = CompoundAccessorIterators<MeshType, CellType>;
    
public:
    // Iterator type template
    template <typename T, typename LocationTag>
    using CellFieldIterator = typename Iterators::template CellFieldIterator<T, LocationTag>;
    
    // Range type template
    template <typename T, typename LocationTag>
    using CellFieldRange = typename Iterators::template CellFieldRange<T, LocationTag>;
    
    // Range-based iteration
    template <typename T, typename LocationTag>
    CellFieldRange<T, LocationTag> cellFields(MeshType& mesh, RegionMask regionMask, FieldMask fieldMask) {
        return CellFieldRange<T, LocationTag>(mesh, regionMask, fieldMask);
    }
    
    template <typename T, typename LocationTag>
    CellFieldRange<T, LocationTag> cellFields(MeshType& mesh, RegionID regionID, FieldID fieldID) {
        RegionMask regionMask = mesh.getRegionMask(regionID);
        FieldMask fieldMask = mesh.getFieldMask(fieldID);
        return CellFieldRange<T, LocationTag>(mesh, regionMask, fieldMask);
    }
    
    template <typename T, typename LocationTag>
    CellFieldRange<T, LocationTag> cellFields(
        MeshType& mesh,
        const std::vector<RegionID>& regionIDs, 
        const std::vector<FieldID>& fieldIDs) {
        
        // Create combined masks
        RegionMask regionMask = 0;
        for (RegionID id : regionIDs) {
            regionMask |= mesh.getRegionMask(id);
        }
        
        FieldMask fieldMask = 0;
        for (FieldID id : fieldIDs) {
            fieldMask |= mesh.getFieldMask(id);
        }
        
        return CellFieldRange<T, LocationTag>(mesh, regionMask, fieldMask);
    }
};

} // namespace CompoundAccessorExtension

/**
 * Mixins for actual accessor classes
 * 
 * These templates can be used with the Curiously Recurring Template Pattern (CRTP)
 * to add iterator functionality to the accessor classes.
 */

// Mixin for MeshAccessor
template <typename MeshType, typename CellType, typename Derived>
class MeshAccessorMixin {
private:
    using Extension = MeshAccessorExtension::Extension<MeshType, CellType>;
    Extension m_extension;
    
    // Helper to get the mesh reference
    MeshType& getMesh() {
        return static_cast<Derived*>(this)->mesh();
    }
    
public:
    // Iterator type aliases
    using CellIterator = typename Extension::CellIterator;
    using InteriorCellIterator = typename Extension::InteriorCellIterator;
    using BoundaryCellIterator = typename Extension::BoundaryCellIterator;
    using BlockCellIterator = typename Extension::BlockCellIterator;
    
    // Range type aliases
    using CellRange = typename Extension::CellRange;
    using InteriorCellRange = typename Extension::InteriorCellRange;
    using BoundaryCellRange = typename Extension::BoundaryCellRange;
    using BlockCellRange = typename Extension::BlockCellRange;
    
    // Iterator access
    CellIterator begin() { return m_extension.begin(getMesh()); }
    CellIterator end() { return m_extension.end(getMesh()); }
    
    // Range access
    CellRange cells() { return m_extension.cells(getMesh()); }
    InteriorCellRange interiorCells() { return m_extension.interiorCells(getMesh()); }
    BoundaryCellRange boundaryCells() { return m_extension.boundaryCells(getMesh()); }
    BlockCellRange blockCells(int blockSizeX = 16, int blockSizeY = 16) {
        return m_extension.blockCells(getMesh(), blockSizeX, blockSizeY);
    }
};

// Mixin for RegionAccessor
template <typename MeshType, typename CellType, typename Derived>
class RegionAccessorMixin {
private:
    using Extension = RegionAccessorExtension::Extension<MeshType, CellType>;
    Extension m_extension;
    
    // Helper to get the mesh reference
    MeshType& getMesh() {
        return static_cast<Derived*>(this)->mesh();
    }
    
public:
    // Iterator type aliases
    using RegionCellIterator = typename Extension::RegionCellIterator;
    
    // Range type aliases
    using RegionCellRange = typename Extension::RegionCellRange;
    
    // Range access
    RegionCellRange cells(RegionMask regionMask) {
        return m_extension.cells(getMesh(), regionMask);
    }
    
    RegionCellRange cells(RegionID regionID) {
        return m_extension.cells(getMesh(), regionID);
    }
};

// Mixin for FieldAccessor
template <typename MeshType, typename Derived>
class FieldAccessorMixin {
private:
    using Extension = FieldAccessorExtension::Extension<MeshType>;
    Extension m_extension;
    
    // Helper to get the mesh reference
    MeshType& getMesh() {
        return static_cast<Derived*>(this)->mesh();
    }
    
    // Helper to get self reference
    Derived& getSelf() {
        return *static_cast<Derived*>(this);
    }
    
public:
    // Iterator type template
    template <typename T, typename LocationTag>
    using FieldIterator = typename Extension::template FieldIterator<T, LocationTag>;
    
    // Range type template
    template <typename T, typename LocationTag>
    using FieldRange = typename Extension::template FieldRange<T, LocationTag>;
    
    // Range access
    template <typename T, typename LocationTag>
    FieldRange<T, LocationTag> fields(FieldMask fieldMask) {
        return m_extension.template fields<T, LocationTag>(getMesh(), fieldMask);
    }
    
    template <typename T, typename LocationTag>
    FieldRange<T, LocationTag> fields(const std::vector<FieldID>& fieldIDs) {
        return m_extension.template fields<T, LocationTag>(getMesh(), fieldIDs, getSelf());
    }
};

// Mixin for CompoundAccessor
template <typename MeshType, typename CellType, typename Derived>
class CompoundAccessorMixin {
private:
    using Extension = CompoundAccessorExtension::Extension<MeshType, CellType>;
    Extension m_extension;
    
    // Helper to get the mesh reference
    MeshType& getMesh() {
        return static_cast<Derived*>(this)->mesh();
    }
    
public:
    // Iterator type template
    template <typename T, typename LocationTag>
    using CellFieldIterator = typename Extension::template CellFieldIterator<T, LocationTag>;
    
    // Range type template
    template <typename T, typename LocationTag>
    using CellFieldRange = typename Extension::template CellFieldRange<T, LocationTag>;
    
    // Range access
    template <typename T, typename LocationTag>
    CellFieldRange<T, LocationTag> cellFields(RegionMask regionMask, FieldMask fieldMask) {
        return m_extension.template cellFields<T, LocationTag>(getMesh(), regionMask, fieldMask);
    }
    
    template <typename T, typename LocationTag>
    CellFieldRange<T, LocationTag> cellFields(RegionID regionID, FieldID fieldID) {
        return m_extension.template cellFields<T, LocationTag>(getMesh(), regionID, fieldID);
    }
    
    template <typename T, typename LocationTag>
    CellFieldRange<T, LocationTag> cellFields(
        const std::vector<RegionID>& regionIDs, 
        const std::vector<FieldID>& fieldIDs) {
        return m_extension.template cellFields<T, LocationTag>(getMesh(), regionIDs, fieldIDs);
    }
};

/**
 * Usage example with CRTP mixins:
 * 
 * // Define your accessor class with the mixin
 * template <typename MeshType, typename CellType>
 * class MeshAccessor : public MeshAccessorMixin<MeshType, CellType, MeshAccessor<MeshType, CellType>> {
 *     // ... existing implementation ...
 * };
 * 
 * // Or add iterator support through inheritance
 * template <typename MeshType, typename CellType>
 * class IterableMeshAccessor : 
 *     public MeshAccessor<MeshType, CellType>,
 *     public MeshAccessorMixin<MeshType, CellType, IterableMeshAccessor<MeshType, CellType>> {
 *     // ... implementation ...
 * };
 */

#endif // ACCESSOR_EXTENSIONS_H


