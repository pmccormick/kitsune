/**
 * @file AccessorIterators.h
 * @brief Iterator interfaces for Accessor classes
 * 
 * This file defines iterator interfaces for the various accessor classes
 * (MeshAccessor, RegionAccessor, FieldAccessor, CompoundAccessor).
 * These iterators build on the Field iterators to provide consistent and
 * efficient traversal patterns for mesh data.
 */

#ifndef FIELD_ACCESSOR_ITERATORS_H
#define FIELD_ACCESSOR_ITERATORS_H

#include "AccessorIteratorsCommon.h"
#include <functional>


template <typename MeshType>
class FieldAccessorIterators {
public:
    /**
     * @brief Iterator for a specific field or group of fields
     * 
     * This template class provides an iterator that can be specialized
     * for different field types and location tags.
     * 
     * @tparam T Field data type
     * @tparam LocationTag Field location tag
     */
    template <typename T, typename LocationTag>
    class FieldIterator {
    public:
        // STL iterator type traits
        using iterator_category = std::forward_iterator_tag;
        using value_type = Field<T, LocationTag>*;
        using difference_type = std::ptrdiff_t;
        using pointer = Field<T, LocationTag>**;
        using reference = Field<T, LocationTag>*&;
        
        /**
         * @brief Construct a new Field Iterator
         * 
         * @param mesh Reference to the mesh
         * @param fieldMask Field mask
         * @param fieldID Current field ID
         */
        FieldIterator(MeshType& mesh, FieldMask fieldMask, FieldID fieldID = 0)
            : m_mesh(mesh), m_fieldMask(fieldMask), m_fieldID(fieldID) {
            // Find first matching field if necessary
            if (fieldID == 0) {
                findNextField();
            }
        }
        
        // Core iterator operations
        value_type operator*() {
            try {
                return &m_mesh.template getField<T, LocationTag>(m_fieldID);
            } catch (const std::exception&) {
                // This should never happen if findNextField works correctly
                return nullptr;
            }
        }
        
        FieldIterator& operator++() {
            ++m_fieldID;
            findNextField();
            return *this;
        }
        
        FieldIterator operator++(int) {
            FieldIterator tmp = *this;
            ++(*this);
            return tmp;
        }
        
        // Comparison
        bool operator==(const FieldIterator& other) const {
            return m_fieldID == other.m_fieldID;
        }
        
        bool operator!=(const FieldIterator& other) const {
            return m_fieldID != other.m_fieldID;
        }
        
        // Current field ID
        FieldID id() const { return m_fieldID; }
        
    private:
        void findNextField() {
            // Maximum possible field ID
            const FieldID maxFieldID = sizeof(FieldMask) * 8;
            
            while (m_fieldID < maxFieldID) {
                // Check if field is in mask and exists with correct type
                FieldMask fieldBit = (FieldMask(1) << m_fieldID);
                
                if ((m_fieldMask & fieldBit) != 0 && m_mesh.hasField(m_fieldID)) {
                    try {
                        // Try to get field with correct type
                        m_mesh.template getField<T, LocationTag>(m_fieldID);
                        return; // Field found
                    } catch (const std::exception&) {
                        // Field exists but has wrong type, continue
                    }
                }
                
                ++m_fieldID;
            }
        }
        
        MeshType& m_mesh;
        FieldMask m_fieldMask;
        FieldID m_fieldID;
    };
    
    /**
     * @brief Range class for fields matching a mask
     * 
     * @tparam T Field data type
     * @tparam LocationTag Field location tag
     */
    template <typename T, typename LocationTag>
    class FieldRange {
    public:
        using iterator = FieldIterator<T, LocationTag>;
        
        FieldRange(MeshType& mesh, FieldMask fieldMask) 
            : m_mesh(mesh), m_fieldMask(fieldMask) {}
        
        iterator begin() { return iterator(m_mesh, m_fieldMask); }
        iterator end() { 
            const FieldID maxFieldID = sizeof(FieldMask) * 8;
            return iterator(m_mesh, m_fieldMask, maxFieldID); 
        }
        
    private:
        MeshType& m_mesh;
        FieldMask m_fieldMask;
    };
};


    template <typename T, typename LocationTag>
    class FieldIterator {
    public:
        // STL iterator type traits
        using iterator_category = std::forward_iterator_tag;
        using value_type = Field<T, LocationTag>*;
        using difference_type = std::ptrdiff_t;
        using pointer = Field<T, LocationTag>**;
        using reference = Field<T, LocationTag>*&;
        
        /**
         * @brief Construct a new Field Iterator
         * 
         * @param mesh Reference to the mesh
         * @param fieldMask Field mask
         * @param fieldID Current field ID
         */
        FieldIterator(MeshType& mesh, FieldMask fieldMask, FieldID fieldID = 0)
            : m_mesh(mesh), m_fieldMask(fieldMask), m_fieldID(fieldID) {
            // Find first matching field if necessary
            if (fieldID == 0) {
                findNextField();
            }
        }
        
        // Core iterator operations
        value_type operator*() {
            try {
                return &m_mesh.template getField<T, LocationTag>(m_fieldID);
            } catch (const std::exception&) {
                // This should never happen if findNextField works correctly
                return nullptr;
            }
        }
        
        FieldIterator& operator++() {
            ++m_fieldID;
            findNextField();
            return *this;
        }
        
        FieldIterator operator++(int) {
            FieldIterator tmp = *this;
            ++(*this);
            return tmp;
        }
        
        // Comparison
        bool operator==(const FieldIterator& other) const {
            return m_fieldID == other.m_fieldID;
        }
        
        bool operator!=(const FieldIterator& other) const {
            return m_fieldID != other.m_fieldID;
        }
        
        // Current field ID
        FieldID id() const { return m_fieldID; }
        
    private:
        void findNextField() {
            // Maximum possible field ID
            const FieldID maxFieldID = sizeof(FieldMask) * 8;
            
            while (m_fieldID < maxFieldID) {
                // Check if field is in mask and exists with correct type
                FieldMask fieldBit = (FieldMask(1) << m_fieldID);
                
                if ((m_fieldMask & fieldBit) != 0 && m_mesh.hasField(m_fieldID)) {
                    try {
                        // Try to get field with correct type
                        m_mesh.template getField<T, LocationTag>(m_fieldID);
                        return; // Field found
                    } catch (const std::exception&) {
                        // Field exists but has wrong type, continue
                    }
                }
                
                ++m_fieldID;
            }
        }
        
        MeshType& m_mesh;
        FieldMask m_fieldMask;
        FieldID m_fieldID;
    };


    template <typename T, typename LocationTag>
    class FieldRange {
    public:
        using iterator = FieldIterator<T, LocationTag>;
        
        FieldRange(MeshType& mesh, FieldMask fieldMask) 
            : m_mesh(mesh), m_fieldMask(fieldMask) {}
        
        iterator begin() { return iterator(m_mesh, m_fieldMask); }
        iterator end() { 
            const FieldID maxFieldID = sizeof(FieldMask) * 8;
            return iterator(m_mesh, m_fieldMask, maxFieldID); 
        }
        
    private:
        MeshType& m_mesh;
        FieldMask m_fieldMask;
    };


    template <typename T, typename LocationTag>
    class CellFieldIterator {
    public:
        // STL iterator type traits
        using iterator_category = std::forward_iterator_tag;
        using value_type = std::pair<CellType*, Field<T, LocationTag>*>;
        using difference_type = std::ptrdiff_t;
        using pointer = value_type*;
        using reference = value_type&;
        
        /**
         * @brief Construct a new Cell Field Iterator
         * 
         * @param mesh Reference to the mesh
         * @param regionMask Region mask
         * @param fieldMask Field mask
         * @param i Initial i-index
         * @param j Initial j-index
         * @param fieldID Initial field ID
         */
        CellFieldIterator(MeshType& mesh, 
                         RegionMask regionMask, 
                         FieldMask fieldMask,
                         int i = 0, int j = 0,
                         FieldID fieldID = 0)
            : m_mesh(mesh), 
              m_regionMask(regionMask), 
              m_fieldMask(fieldMask),
              m_i(i), m_j(j), 
              m_fieldID(fieldID),
              m_currentCell(nullptr),
              m_currentField(nullptr) {
            // Find first valid cell-field pair if necessary
            if (i == 0 && j == 0 && fieldID == 0) {
                findNextPair();
            }
        }
        
        // Core iterator operations
        value_type operator*() const { 
            return std::make_pair(m_currentCell, m_currentField); 
        }
        
        CellFieldIterator& operator++() {
            // Try next field for same cell
            ++m_fieldID;
            
            // If we've gone through all fields, move to next cell
            if (!findNextField()) {
                m_fieldID = 0;
                ++m_i;
                if (m_i >= m_mesh.nx()) {
                    m_i = 0;
                    ++m_j;
                }
                
                findNextPair();
            }
            
            return *this;
        }
        
        CellFieldIterator operator++(int) {
            CellFieldIterator tmp = *this;
            ++(*this);
            return tmp;
        }
        
        // Comparison
        bool operator==(const CellFieldIterator& other) const {
            return m_i == other.m_i && 
                   m_j == other.m_j && 
                   m_fieldID == other.m_fieldID;
        }
        
        bool operator!=(const CellFieldIterator& other) const {
            return !(*this == other);
        }
        
        // Current indices
        int i() const { return m_i; }
        int j() const { return m_j; }
        FieldID fieldID() const { return m_fieldID; }
        
    private:
        bool findNextField() {
            // Maximum possible field ID
            const FieldID maxFieldID = sizeof(FieldMask) * 8;
            
            while (m_fieldID < maxFieldID) {
                // Check if field is in mask and exists with correct type
                FieldMask fieldBit = (FieldMask(1) << m_fieldID);
                
                if ((m_fieldMask & fieldBit) != 0 && m_mesh.hasField(m_fieldID)) {
                    try {
                        // Try to get field with correct type
                        m_currentField = &m_mesh.template getField<T, LocationTag>(m_fieldID);
                        return true; // Field found
                    } catch (const std::exception&) {
                        // Field exists but has wrong type, continue
                    }
                }
                
                ++m_fieldID;
            }
            
            m_currentField = nullptr;
            return false;
        }
        
        bool isCellInRegion(const CellType* cell, RegionMask mask) const {
            // This is simplified and would need to match the actual implementation
            // in RegionAccessor::isCellInRegion
            int linearIndex = cell->linearIndex();
            linearIndex %= (sizeof(RegionMask) * 8);
            RegionMask cellBit = RegionMask(1) << linearIndex;
            return (mask & cellBit) != 0;
        }
        
        void findNextPair() {
            // Find next cell in region
            while (m_j < m_mesh.ny()) {
                m_currentCell = m_mesh.getTypedCell(m_i, m_j);
                
                if (m_currentCell && isCellInRegion(m_currentCell, m_regionMask)) {
                    // Found cell in region, now find matching field
                    if (findNextField()) {
                        return; // Found a valid cell-field pair
                    }
                }
                
                // No valid field for this cell, move to next cell
                m_fieldID = 0;
                ++m_i;
                if (m_i >= m_mesh.nx()) {
                    m_i = 0;
                    ++m_j;
                }
            }
            
            // No more valid pairs
            m_currentCell = nullptr;
            m_currentField = nullptr;
        }
        
        MeshType& m_mesh;
        RegionMask m_regionMask;
        FieldMask m_fieldMask;
        int m_i, m_j;
        FieldID m_fieldID;
        CellType* m_currentCell;
        Field<T, LocationTag>* m_currentField;
    };


    template <typename T, typename LocationTag>
    class CellFieldRange {
    public:
        using iterator = CellFieldIterator<T, LocationTag>;
        
        CellFieldRange(MeshType& mesh, RegionMask regionMask, FieldMask fieldMask) 
            : m_mesh(mesh), m_regionMask(regionMask), m_fieldMask(fieldMask) {}
        
        iterator begin() { 
            return iterator(m_mesh, m_regionMask, m_fieldMask); 
        }
        
        iterator end() { 
            return iterator(m_mesh, m_regionMask, m_fieldMask, 0, m_mesh.ny(), 0); 
        }
        
    private:
        MeshType& m_mesh;
        RegionMask m_regionMask;
        FieldMask m_fieldMask;
    };

#endif // FIELD_ACCESSOR_ITERATORS_H
