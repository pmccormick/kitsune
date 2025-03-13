#ifndef MOCK_REGION_H
#define MOCK_REGION_H

#include "BitArray.h"
#include "Cell.h"
#include <unordered_set>
#include <vector>
#include <string>
#include <memory>
#include <functional>

namespace mock {

/**
 * @brief Mock implementation of Region for testing
 * 
 * This class provides a simplified Region implementation
 * that can be used for testing without depending on the actual Region class.
 */
class MockRegion {
public:
    /**
     * @brief Storage mode enumeration matching the real Region class
     */
    enum class StorageMode {
        CELL_SET = 0,
        BIT_ARRAY = 1,
        DYNAMIC = 2
    };
    
    /**
     * @brief Construct a new Mock Region with bit array storage
     * 
     * @param id Region ID
     * @param name Region name
     * @param meshSize Total number of cells in mesh
     */
    MockRegion(uint32_t id, const std::string& name, size_t meshSize)
        : m_id(id), 
          m_name(name), 
          m_mode(StorageMode::BIT_ARRAY),
          m_bitArray(meshSize, false),
          m_meshSize(meshSize) {}
    
    /**
     * @brief Check if a cell is in this region
     * 
     * @param cell Pointer to cell to check
     * @return true if cell is in region
     */
    bool contains(const mesh::Cell* cell) const {
        if (!cell || !cell->isValid()) return false;
        
        int linearIndex = cell->index();
        return containsIndex(linearIndex);
    }
    
    /**
     * @brief Check if a linear index is in this region
     * 
     * @param linearIndex Linear index to check
     * @return true if index is in region
     */
    bool containsIndex(int linearIndex) const {
        if (linearIndex < 0 || linearIndex >= static_cast<int>(m_meshSize)) {
            return false;
        }
        
        if (m_mode == StorageMode::BIT_ARRAY) {
            return m_bitArray.get(linearIndex);
        } else if (m_mode == StorageMode::CELL_SET) {
            return m_cellIndices.find(linearIndex) != m_cellIndices.end();
        } else if (m_mode == StorageMode::DYNAMIC) {
            return m_predicate ? m_predicate(linearIndex) : false;
        }
        
        return false;
    }
    
    /**
     * @brief Add a cell to the region
     * 
     * @param cell Pointer to cell to add
     */
    void addCell(const mesh::Cell* cell) {
        if (!cell || !cell->isValid()) return;
        
        int linearIndex = cell->index();
        addCellIndex(linearIndex);
    }
    
    /**
     * @brief Add a linear index to the region
     * 
     * @param linearIndex Linear index to add
     */
    void addCellIndex(int linearIndex) {
        if (linearIndex < 0 || linearIndex >= static_cast<int>(m_meshSize)) {
            return;
        }
        
        if (m_mode == StorageMode::BIT_ARRAY) {
            m_bitArray.set(linearIndex, true);
        } else if (m_mode == StorageMode::CELL_SET) {
            m_cellIndices.insert(linearIndex);
        }
    }
    
    /**
     * @brief Remove a cell from the region
     * 
     * @param cell Pointer to cell to remove
     */
    void removeCell(const mesh::Cell* cell) {
        if (!cell || !cell->isValid()) return;
        
        int linearIndex = cell->index();
        removeCellIndex(linearIndex);
    }
    
    /**
     * @brief Remove a linear index from the region
     * 
     * @param linearIndex Linear index to remove
     */
    void removeCellIndex(int linearIndex) {
        if (linearIndex < 0 || linearIndex >= static_cast<int>(m_meshSize)) {
            return;
        }
        
        if (m_mode == StorageMode::BIT_ARRAY) {
            m_bitArray.set(linearIndex, false);
        } else if (m_mode == StorageMode::CELL_SET) {
            m_cellIndices.erase(linearIndex);
        }
    }
    
    /**
     * @brief Set storage mode
     * 
     * @param mode New storage mode
     */
    void setStorageMode(StorageMode mode) {
        if (mode == m_mode) return;
        
        if (mode == StorageMode::BIT_ARRAY && m_mode == StorageMode::CELL_SET) {
            // Convert cell set to bit array
            m_bitArray = BitArray(m_meshSize, false);
            for (int idx : m_cellIndices) {
                m_bitArray.set(idx, true);
            }
            m_cellIndices.clear();
        } else if (mode == StorageMode::CELL_SET && m_mode == StorageMode::BIT_ARRAY) {
            // Convert bit array to cell set
            m_cellIndices.clear();
            for (size_t idx = m_bitArray.findFirst(); idx < m_bitArray.size(); idx = m_bitArray.findNext(idx)) {
                m_cellIndices.insert(static_cast<int>(idx));
            }
            m_bitArray = BitArray(0);  // Clear bit array
        }
        
        m_mode = mode;
    }
    
    /**
     * @brief Get current storage mode
     * 
     * @return StorageMode Current storage mode
     */
    StorageMode getStorageMode() const {
        return m_mode;
    }
    
    /**
     * @brief Get number of cells in the region
     * 
     * @return size_t Number of cells
     */
    size_t size() const {
        if (m_mode == StorageMode::BIT_ARRAY) {
            return m_bitArray.count();
        } else if (m_mode == StorageMode::CELL_SET) {
            return m_cellIndices.size();
        } else if (m_mode == StorageMode::DYNAMIC && m_predicate) {
            // Count cells that match the predicate (expensive!)
            size_t count = 0;
            for (size_t i = 0; i < m_meshSize; ++i) {
                if (m_predicate(static_cast<int>(i))) {
                    ++count;
                }
            }
            return count;
        }
        
        return 0;
    }
    
    /**
     * @brief Get region ID
     * 
     * @return uint32_t Region ID
     */
    uint32_t getId() const {
        return m_id;
    }
    
    /**
     * @brief Get region name
     * 
     * @return const std::string& Region name
     */
    const std::string& getName() const {
        return m_name;
    }
    
    /**
     * @brief Get mesh size
     * 
     * @return size_t Total number of cells in mesh
     */
    size_t getMeshSize() const {
        return m_meshSize;
    }
    
    /**
     * @brief Set a predicate function for dynamic mode
     * 
     * @param predicate Function determining cell membership
     */
    void setPredicate(std::function<bool(int)> predicate) {
        m_predicate = std::move(predicate);
        if (predicate) {
            m_mode = StorageMode::DYNAMIC;
        }
    }
    
    /**
     * @brief Get bit array for testing
     * 
     * @return const BitArray& Bit array representation
     */
    const BitArray& getBitArray() const {
        return m_bitArray;
    }
    
    /**
     * @brief Get cell indices for testing
     * 
     * @return const std::unordered_set<int>& Cell indices set
     */
    const std::unordered_set<int>& getCellIndices() const {
        return m_cellIndices;
    }
    
    /**
     * @brief Clear all cells from the region
     */
    void clear() {
        if (m_mode == StorageMode::BIT_ARRAY) {
            m_bitArray.clear();
        } else if (m_mode == StorageMode::CELL_SET) {
            m_cellIndices.clear();
        }
    }
    
private:
    uint32_t m_id;  ///< Region ID
    std::string m_name;  ///< Region name
    StorageMode m_mode;  ///< Current storage mode
    BitArray m_bitArray;  ///< Bit array representation
    std::unordered_set<int> m_cellIndices;  ///< Cell set representation
    size_t m_meshSize;  ///< Total number of cells in mesh
    std::function<bool(int)> m_predicate;  ///< Predicate function for dynamic mode
};

} // namespace mock

#endif // MOCK_REGION_H
