/**
 * @file RegionAccessor.h
 * @brief Provides efficient accessor patterns for region-based field operations
 * 
 * This file defines the RegionAccessor class which provides optimized
 * access patterns for field data within specific regions. This allows for
 * efficient implementation of operations that only need to be applied to
 * a subset of the mesh.
 */

#ifndef REGION_ACCESSOR_H
#define REGION_ACCESSOR_H

#include "Region.h"
#include "Cell.h"
#include "Mesh.h"
#include "Field.h"
#include "RegionUtils.h"
#include <functional>
#include <vector>
#include <memory>
#include <concepts>

namespace mesh {

/**
 * @brief Accessor for efficient region-based operations
 * 
 * The RegionAccessor provides optimized patterns for working with field data
 * within specific regions. It pre-computes access patterns based on region
 * characteristics to minimize overhead during operations.
 * 
 * @tparam MeshT Type of mesh
 */
template <typename MeshT>
class RegionAccessor {
public:
    using cell_type = typename MeshT::cell_type;

    /**
     * @brief Construct a new Region Accessor
     * 
     * @param mesh Reference to the mesh
     * @param region Reference to the region
     */
    RegionAccessor(MeshT& mesh, const Region& region)
        : m_mesh(mesh), m_region(region) {
        // Pre-compute cell list for efficient access
        refreshCellList();
    }

    /**
     * @brief Apply a function to each cell in the region
     * 
     * @tparam FuncT Type of function to apply
     * @param func Function to apply to each cell
     */
    template <typename FuncT>
    requires std::invocable<FuncT, cell_type*>
    void forEachCell(FuncT func) const {
        for (cell_type* cell : m_cells) {
            func(cell);
        }
    }

    /**
     * @brief Apply a function to each cell in the region with position information
     * 
     * @tparam LengthUnitT Type of length unit for position
     * @tparam FuncT Type of function to apply
     * @param func Function to apply to each cell with its position
     */
    template <typename LengthUnitT, typename FuncT>
    requires std::invocable<FuncT, cell_type*, physics::Vector2D<LengthUnitT>>
    void forEachCellWithPosition(FuncT func) const {
        m_region.template forEachCellWithPosition<LengthUnitT>([&](mesh::Cell* baseCell, const physics::Vector2D<LengthUnitT>& pos) {
            // Cast to the specific cell type
            cell_type* cell = static_cast<cell_type*>(baseCell);
            func(cell, pos);
        });
    }

    /**
     * @brief Get a field value for each cell in the region
     * 
     * @tparam ValueT Field value type
     * @tparam UnitT Field unit type
     * @tparam LocationTag Field location tag
     * @param field Field to access
     * @return Vector of values from the field for cells in the region
     */
    template <typename ValueT, typename UnitT, typename LocationTag = mesh::CellCenterTag>
    std::vector<ValueT> getFieldValues(const mesh::Field<ValueT, UnitT, LocationTag>& field) const {
        std::vector<ValueT> values;
        values.reserve(m_cells.size());

        for (cell_type* cell : m_cells) {
            int i = cell->i();
            int j = cell->j();
            values.push_back(field(i, j));
        }

        return values;
    }

    /**
     * @brief Get average field value for cells in the region
     * 
     * @tparam ValueT Field value type
     * @tparam UnitT Field unit type
     * @tparam LocationTag Field location tag
     * @param field Field to access
     * @return Average value from the field for cells in the region
     */
    template <typename ValueT, typename UnitT, typename LocationTag = mesh::CellCenterTag>
    ValueT getAverageFieldValue(const mesh::Field<ValueT, UnitT, LocationTag>& field) const {
        if (m_cells.empty()) {
            return ValueT(0);
        }

        ValueT sum = 0;
        for (cell_type* cell : m_cells) {
            int i = cell->i();
            int j = cell->j();
            sum += field(i, j);
        }

        return sum / static_cast<ValueT>(m_cells.size());
    }

    /**
     * @brief Set field values for all cells in the region
     * 
     * @tparam ValueT Field value type
     * @tparam UnitT Field unit type
     * @tparam LocationTag Field location tag
     * @param field Field to modify
     * @param value Value to set
     */
    template <typename ValueT, typename UnitT, typename LocationTag = mesh::CellCenterTag>
    void setFieldValue(mesh::Field<ValueT, UnitT, LocationTag>& field, const ValueT& value) const {
        for (cell_type* cell : m_cells) {
            int i = cell->i();
            int j = cell->j();
            field(i, j) = value;
        }
    }

    /**
     * @brief Set field values using a function for all cells in the region
     * 
     * @tparam ValueT Field value type
     * @tparam UnitT Field unit type
     * @tparam LocationTag Field location tag
     * @tparam FuncT Function type
     * @param field Field to modify
     * @param func Function that returns the value for each cell
     */
    template <typename ValueT, typename UnitT, typename LocationTag = mesh::CellCenterTag, typename FuncT>
    requires std::invocable<FuncT, cell_type*>
    void setFieldValues(mesh::Field<ValueT, UnitT, LocationTag>& field, FuncT func) const {
        for (cell_type* cell : m_cells) {
            int i = cell->i();
            int j = cell->j();
            field(i, j) = func(cell);
        }
    }

    /**
     * @brief Set field values using a unit-aware function for all cells in the region
     * 
     * @tparam ValueT Field value type
     * @tparam UnitT Field unit type
     * @tparam InputUnitT Input unit type
     * @tparam LocationTag Field location tag
     * @tparam FuncT Function type
     * @param field Field to modify
     * @param func Function that returns a unit-aware value for each cell
     */
    template <typename ValueT, typename UnitT, typename InputUnitT, typename LocationTag = mesh::CellCenterTag, typename FuncT>
    requires std::invocable<FuncT, cell_type*> && 
             units::UnitType<InputUnitT> && 
             units::SameDimension<UnitT, InputUnitT>
    void setFieldValuesWithUnits(mesh::Field<ValueT, UnitT, LocationTag>& field, FuncT func) const {
        for (cell_type* cell : m_cells) {
            int i = cell->i();
            int j = cell->j();
            InputUnitT value = func(cell);
            field.template setValueAs<InputUnitT>(i, j, value);
        }
    }

    /**
     * @brief Get number of cells in the region
     * 
     * @return Size of the region
     */
    size_t size() const {
        return m_cells.size();
    }

    /**
     * @brief Get reference to the region
     * 
     * @return Reference to the region
     */
    const Region& region() const {
        return m_region;
    }

    /**
     * @brief Refresh the internal cell list
     * 
     * This needs to be called if the region or mesh changes.
     */
    void refreshCellList() {
        m_cells = getCellsInRegion(m_mesh, m_region);
    }

private:
    MeshT& m_mesh;                   ///< Reference to the mesh
    const Region& m_region;          ///< Reference to the region
    std::vector<cell_type*> m_cells; ///< Cached list of cells in the region
};

/**
 * @brief Create a region accessor
 * 
 * Helper function for creating region accessors with template parameter deduction.
 * 
 * @tparam MeshT Type of mesh
 * @param mesh Reference to the mesh
 * @param region Reference to the region
 * @return RegionAccessor for the mesh and region
 */
template <typename MeshT>
RegionAccessor<MeshT> createRegionAccessor(MeshT& mesh, const Region& region) {
    return RegionAccessor<MeshT>(mesh, region);
}

}// namespace 
#endif // REGION_ACCESSOR_H

