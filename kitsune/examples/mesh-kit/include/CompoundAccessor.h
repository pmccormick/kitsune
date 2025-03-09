/**
 * @file CompoundAccessor.h
 * @brief Accessor for combined field and region operations
 * 
 * This header defines an accessor for performing operations that
 * involve both field selection and region filtering.
 */

#ifndef COMPOUND_ACCESSOR_H
#define COMPOUND_ACCESSOR_H

#include "Mesh.h"
#include "RegionAccessor.h"
#include "FieldAccessor.h"
#include <functional>
#include <vector>

/**
 * @brief Accessor for combined field and region operations
 * 
 * Provides operations that involve both field selection and
 * region filtering for more complex computational patterns.
 * 
 * @tparam MeshType The specialized Mesh type
 * @tparam CellType The cell type used by the mesh
 */
template <typename MeshType, typename CellType>
class CompoundAccessor {
public:
    /**
     * @brief Construct a new Compound Accessor
     * 
     * @param mesh Reference to the mesh
     */
    explicit CompoundAccessor(MeshType& mesh) 
        : m_mesh(mesh), 
          m_regionAccessor(mesh), 
          m_fieldAccessor(mesh) {}
    
    /**
     * @brief Apply an operation to a specific field in a specific region
     * 
     * @tparam T Field data type
     * @tparam LocationTag Field location tag
     * @tparam Op Operation type
     * @param regionID Region identifier
     * @param fieldID Field identifier
     * @param op Operation to apply (takes cell and field reference)
     */
    template <typename T, typename LocationTag, typename Op>
    void apply(RegionID regionID, FieldID fieldID, Op op) {
        RegionMask regionMask = m_mesh.getRegionMask(regionID);
        FieldMask fieldMask = m_mesh.getFieldMask(fieldID);
        
        apply<T, LocationTag>(regionMask, fieldMask, op);
    }
    
    /**
     * @brief Apply an operation using bit masks
     * 
     * @tparam T Field data type
     * @tparam LocationTag Field location tag
     * @tparam Op Operation type
     * @param regionMask Region mask
     * @param fieldMask Field mask
     * @param op Operation to apply (takes cell and field reference)
     */
    template <typename T, typename LocationTag, typename Op>
    void apply(RegionMask regionMask, FieldMask fieldMask, Op op) {
        // First ensure all fields exist and match the requested type
        std::vector<Field<T, LocationTag>*> fields;
        
        // Collect all fields that match the mask
        for (FieldID fieldID = 0; fieldID < 64; ++fieldID) {
            FieldMask fieldBit = m_fieldAccessor.getFieldMask(fieldID);
            
            if ((fieldMask & fieldBit) != 0 && m_mesh.hasField(fieldID)) {
                try {
                    Field<T, LocationTag>& field = m_mesh.template getField<T, LocationTag>(fieldID);
                    fields.push_back(&field);
                } catch (const std::exception&) {
                    // Skip fields that don't match the requested type
                    continue;
                }
            }
        }
        
        // If no fields match, return early
        if (fields.empty()) {
            return;
        }
        
        // Apply the operation to cells in the region for each matching field
        m_regionAccessor.forEachCell(regionMask, [&](CellType* cell) {
            for (auto* field : fields) {
                op(cell, *field);
            }
        });
    }
    
    /**
     * @brief Apply an operation to multiple fields in multiple regions
     * 
     * @tparam T Field data type
     * @tparam LocationTag Field location tag
     * @tparam Op Operation type
     * @param regionIDs Vector of region identifiers
     * @param fieldIDs Vector of field identifiers
     * @param op Operation to apply (takes cell and field reference)
     */
    template <typename T, typename LocationTag, typename Op>
    void applyMulti(const std::vector<RegionID>& regionIDs, 
                   const std::vector<FieldID>& fieldIDs, 
                   Op op) {
        // Create combined masks
        RegionMask regionMask = 0;
        for (RegionID id : regionIDs) {
            regionMask |= m_mesh.getRegionMask(id);
        }
        
        FieldMask fieldMask = 0;
        for (FieldID id : fieldIDs) {
            fieldMask |= m_mesh.getFieldMask(id);
        }
        
        // Apply using the combined masks
        apply<T, LocationTag>(regionMask, fieldMask, op);
    }
    
    /**
     * @brief Get access to the region accessor
     * 
     * @return RegionAccessor<MeshType, CellType>& Reference to region accessor
     */
    RegionAccessor<MeshType, CellType>& regions() { return m_regionAccessor; }
    
    /**
     * @brief Get access to the field accessor
     * 
     * @return FieldAccessor<MeshType>& Reference to field accessor
     */
    FieldAccessor<MeshType>& fields() { return m_fieldAccessor; }

private:
    MeshType& m_mesh;  ///< Reference to the mesh
    RegionAccessor<MeshType, CellType> m_regionAccessor;  ///< Region accessor
    FieldAccessor<MeshType> m_fieldAccessor;  ///< Field accessor
};

#endif // COMPOUND_ACCESSOR_H

