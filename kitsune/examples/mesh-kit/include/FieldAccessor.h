/**
 * @file a
 * @brief Accessor for field operations and grouping
 * 
 * This header defines an accessor for performing operations on
 * specific fields or groups of fields using bit-based techniques.
 */

#ifndef FIELD_ACCESSOR_H
#define FIELD_ACCESSOR_H

#include "Mesh.h"
#include <functional>
#include <vector>

/**
 * @brief Accessor for field operations
 * 
 * Enables efficient operations across multiple fields
 * using bitwise field selection.
 * 
 * @tparam MeshType The specialized Mesh type
 */
template <typename MeshType>
class FieldAccessor {
public:
    /**
     * @brief Construct a new Field Accessor
     * 
     * @param mesh Reference to the mesh
     */
    explicit FieldAccessor(MeshType& mesh) : m_mesh(mesh) {}
    
    /**
     * @brief Apply an operation to a specific field
     * 
     * @tparam T Field data type
     * @tparam LocationTag Field location tag
     * @tparam Op Operation type
     * @param fieldID Field identifier
     * @param op Operation to apply
     */
    template <typename T, typename LocationTag, typename Op>
    void applyToField(FieldID fieldID, Op op) {
        if (!m_mesh.hasField(fieldID)) {
            throw std::runtime_error("Field does not exist");
        }
        
        Field<T, LocationTag>& field = m_mesh.template getField<T, LocationTag>(fieldID);
        op(field);
    }
    
    /**
     * @brief Apply an operation to multiple fields
     * 
     * @tparam T Field data type
     * @tparam LocationTag Field location tag
     * @tparam Op Operation type
     * @param fieldMask Bit mask selecting fields
     * @param op Operation to apply
     */
    template <typename T, typename LocationTag, typename Op>
    void applyToFields(FieldMask fieldMask, Op op) {
        // Iterate through all fields and apply the operation if the field is in the mask
        // This is a simplified implementation
        for (FieldID fieldID = 0; fieldID < 64; ++fieldID) {
            FieldMask fieldBit = getFieldBit(fieldID);
            
            if ((fieldMask & fieldBit) != 0 && m_mesh.hasField(fieldID)) {
                try {
                    Field<T, LocationTag>& field = m_mesh.template getField<T, LocationTag>(fieldID);
                    op(field);
                } catch (const std::exception&) {
                    // Skip fields that don't match the requested type
                    continue;
                }
            }
        }
    }
    
    /**
     * @brief Get the field mask for a field ID
     * 
     * @param fieldID Field identifier
     * @return FieldMask Bit mask for the field
     */
    FieldMask getFieldMask(FieldID fieldID) const {
        return m_mesh.getFieldMask(fieldID);
    }
    
    /**
     * @brief Create a field mask from multiple field IDs
     * 
     * @param fieldIDs Vector of field identifiers
     * @return FieldMask Combined mask for all specified fields
     */
    FieldMask createMask(const std::vector<FieldID>& fieldIDs) const {
        FieldMask mask = 0;
        
        for (FieldID fieldID : fieldIDs) {
            mask |= getFieldMask(fieldID);
        }
        
        return mask;
    }
    
    /**
     * @brief Union of two field masks
     * 
     * @param mask1 First field mask
     * @param mask2 Second field mask
     * @return FieldMask Union of the masks
     */
    static FieldMask unionMasks(FieldMask mask1, FieldMask mask2) {
        return mask1 | mask2;
    }
    
    /**
     * @brief Intersection of two field masks
     * 
     * @param mask1 First field mask
     * @param mask2 Second field mask
     * @return FieldMask Intersection of the masks
     */
    static FieldMask intersectMasks(FieldMask mask1, FieldMask mask2) {
        return mask1 & mask2;
    }

private:
    MeshType& m_mesh;  ///< Reference to the mesh
    
    /**
     * @brief Get the bit corresponding to a field in field masks
     * 
     * @param fieldID Field identifier
     * @return FieldMask Bit mask with only the field's bit set
     */
    FieldMask getFieldBit(FieldID fieldID) const {
        // Ensure the field ID is within the available bits
        fieldID %= (sizeof(FieldMask) * 8);
        
        return FieldMask(1) << fieldID;
    }
};

#endif // FIELD_ACCESSOR_H


