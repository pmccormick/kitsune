/**
 * This test focuses on debugging why cells aren't being added to the region
 */

#include "gtest/gtest.h"
#include "CellBase.h"
#include "MeshBase.h"
#include "Region.h"
#include <vector>
#include <functional>
#include <iostream>

// Simple test cell implementation with default constructor
class DebugCell : public CellBase {
public:
    DebugCell() : CellBase(0, 0, nullptr), value(0.0) {}
    
    DebugCell(int i, int j, MeshBase* mesh) : CellBase(i, j, mesh), value(0.0) {}
    
    double value;
};

// Simple test mesh implementation
class DebugMesh : public MeshBase {
public:
    DebugMesh(int nx, int ny) : MeshBase(nx, ny, 1.0, 1.0) {
        // Initialize cells
        m_cells.resize(nx * ny);
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                m_cells[linearIndex(i, j)] = DebugCell(i, j, this);
                m_cells[linearIndex(i, j)].value = i * 10.0 + j;
            }
        }
    }
    
    CellBase* getCell(int i, int j) override {
        if (i < 0 || i >= nx() || j < 0 || j >= ny()) {
            return nullptr;
        }
        return &m_cells[linearIndex(i, j)];
    }
    
    DebugCell* getTypedCell(int i, int j) {
        if (i < 0 || i >= nx() || j < 0 || j >= ny()) {
            return nullptr;
        }
        return &m_cells[linearIndex(i, j)];
    }
    
private:
    std::vector<DebugCell> m_cells;
};

// Test class that inherits from Region to debug the addCellIndex method
class DebugRegion : public Region {
public:
    using Region::Region;
    
    void debugAddCell(int linearIndex) {
        std::cout << "Adding cell with linear index: " << linearIndex << std::endl;
        std::cout << "  Before: size = " << size() << std::endl;
        
        // Call the actual addCellIndex method
        addCellIndex(linearIndex);
        
        std::cout << "  After: size = " << size() << std::endl;
        std::cout << "  Contains cell: " << containsIndex(linearIndex) << std::endl;
    }
};

TEST(RegionDebugSuite, AddCellDebugTest) {
    DebugMesh mesh(10, 10);
    
    // Create a debug region
    auto definition = std::make_shared<PredicateRegion>("TestRegion", 
        [](const CellBase*) { return true; });
    DebugRegion region(1, definition, mesh.nx() * mesh.ny());
    
    std::cout << "Initial region size: " << region.size() << std::endl;
    
    // Add a few cells directly to test addCellIndex
    region.debugAddCell(mesh.linearIndex(5, 0));
    region.debugAddCell(mesh.linearIndex(6, 0));
    region.debugAddCell(mesh.linearIndex(7, 0));
    
    // Verify these cells are in the region
    std::cout << "Final region size: " << region.size() << std::endl;
    
    for (int i = 5; i <= 7; i++) {
        DebugCell* cell = mesh.getTypedCell(i, 0);
        bool contains = region.contains(cell);
        std::cout << "Contains cell (" << i << ",0): " << contains << std::endl;
        EXPECT_TRUE(contains) << "Cell at (" << i << ",0) should be in the region";
    }
}

/**
 * This test directly manipulates a Region to understand the core functionality
 */
TEST(RegionDebugSuite, DirectRegionTest) {
    // Create a region with a simple definition
    auto def = std::make_shared<PredicateRegion>("AlwaysTrue", 
        [](const CellBase*) { return true; });
    Region region(1, def, 100); // ID 1, capacity 100
    
    // Add some cell indices directly
    for (int i = 0; i < 10; i++) {
        int index = i * 10; // Some arbitrary indices
        std::cout << "Adding cell index " << index << std::endl;
        region.addCellIndex(index);
    }
    
    // Print the size
    std::cout << "Region size: " << region.size() << std::endl;
    
    // Check if the indices we added are in the region
    for (int i = 0; i < 10; i++) {
        int index = i * 10;
        bool contains = region.containsIndex(index);
        std::cout << "Contains index " << index << ": " << contains << std::endl;
        EXPECT_TRUE(contains) << "Region should contain index " << index;
    }
    
    // Check some indices that shouldn't be in the region
    for (int i = 1; i < 10; i++) {
        int index = i * 10 - 1; // Indices we didn't add
        bool contains = region.containsIndex(index);
        std::cout << "Contains index " << index << ": " << contains << std::endl;
        EXPECT_FALSE(contains) << "Region should not contain index " << index;
    }
    
    // Test storage mode change
    std::cout << "Current storage mode: " << static_cast<int>(region.getStorageMode()) << std::endl;
    region.setStorageMode(Region::StorageMode::BIT_VECTOR);
    std::cout << "New storage mode: " << static_cast<int>(region.getStorageMode()) << std::endl;
    
    // Check if indices are still in the region after storage mode change
    for (int i = 0; i < 10; i++) {
        int index = i * 10;
        bool contains = region.containsIndex(index);
        std::cout << "Contains index " << index << " after mode change: " << contains << std::endl;
        EXPECT_TRUE(contains) << "Region should still contain index " << index;
    }
}

/**
 * This test checks how cells and regions interact with respect to cell identity
 */
TEST(RegionDebugSuite, CellIdentityTest) {
    class IdentityCell : public CellBase {
    public:
        IdentityCell() : CellBase(0, 0, nullptr) {}
        IdentityCell(int i, int j, MeshBase* mesh) : CellBase(i, j, mesh) {}
        
        // Override linearIndex to debug how it's used by Region
        int linearIndex() const {
            std::cout << "IdentityCell::linearIndex() called for cell at (" << i() << "," << j() << ")" << std::endl;
            if (mesh() == nullptr) {
                std::cout << "  WARNING: mesh() is null!" << std::endl;
                return -1;
            }
            int index = mesh()->linearIndex(i(), j());
            std::cout << "  Returning linearIndex: " << index << std::endl;
            return index;
        }
    };

    class IdentityMesh : public MeshBase {
    public:
        IdentityMesh(int nx, int ny) : MeshBase(nx, ny, 1.0, 1.0) {
            m_cells.resize(nx * ny);
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    m_cells[linearIndex(i, j)] = IdentityCell(i, j, this);
                }
            }
        }
        
        CellBase* getCell(int i, int j) override {
            if (i < 0 || i >= nx() || j < 0 || j >= ny()) {
                return nullptr;
            }
            std::cout << "IdentityMesh::getCell(" << i << "," << j << ") called" << std::endl;
            std::cout << "  Returning cell at linearIndex: " << linearIndex(i, j) << std::endl;
            return &m_cells[linearIndex(i, j)];
        }
        
    private:
        std::vector<IdentityCell> m_cells;
    };

    IdentityMesh mesh(5, 5);
    
    // Create a region
    auto def = std::make_shared<PredicateRegion>("TestRegion", [](const CellBase*) { return true; });
    Region region(1, def, 25);
    
    // Get a cell and add it to the region
    CellBase* cell1 = mesh.getCell(2, 3);
    std::cout << "\nAdding cell (2,3) to region" << std::endl;
    region.addCell(cell1);
    
    // Get the same cell again and check if it's in the region
    CellBase* cell2 = mesh.getCell(2, 3);
    std::cout << "\nChecking if cell (2,3) is in region" << std::endl;
    bool contains = region.contains(cell2);
    
    std::cout << "Cell1 linearIndex: " << cell1->linearIndex() << std::endl;
    std::cout << "Cell2 linearIndex: " << cell2->linearIndex() << std::endl;
    std::cout << "Region contains cell: " << contains << std::endl;
    
    EXPECT_TRUE(contains) << "Region should contain the cell we just added";
    
    // Try with a cell we didn't add
    CellBase* cell3 = mesh.getCell(1, 1);
    std::cout << "\nChecking if cell (1,1) is in region" << std::endl;
    bool shouldBeFalse = region.contains(cell3);
    
    std::cout << "Cell3 linearIndex: " << cell3->linearIndex() << std::endl;
    std::cout << "Region contains cell: " << shouldBeFalse << std::endl;
    
    EXPECT_FALSE(shouldBeFalse) << "Region should not contain a cell we didn't add";
}

/**
 * A focused test for the predicate region specifically
 */
TEST(RegionDebugSuite, PredicateRegionTest) {
    DebugMesh mesh(10, 10);
    
    // Create a region based on cell values > 50
    auto def = std::make_shared<PredicateRegion>("HighValueRegion", 
        [](const CellBase* cell) { 
            const DebugCell* dcell = static_cast<const DebugCell*>(cell);
            bool result = dcell->value > 50.0;
            std::cout << "Predicate for cell (" << dcell->i() << "," << dcell->j() 
                      << ") with value " << dcell->value << " returned " << result << std::endl;
            return result;
        });
    
    Region region(1, def, mesh.nx() * mesh.ny());
    
    // Explicitly add cells that satisfy the predicate
    int addedCount = 0;
    for (int j = 0; j < mesh.ny(); ++j) {
        for (int i = 0; i < mesh.nx(); ++i) {
            DebugCell* cell = mesh.getTypedCell(i, j);
            if (cell->value > 50.0) {
                std::cout << "Adding cell (" << i << "," << j << ") with value " << cell->value << std::endl;
                region.addCell(cell);
                addedCount++;
            }
        }
    }
    
    std::cout << "Added " << addedCount << " cells to region" << std::endl;
    std::cout << "Region size: " << region.size() << std::endl;
    
    // Check if cells with values > 50 are in the region
    int successCount = 0;
    int failCount = 0;
    for (int j = 0; j < mesh.ny(); ++j) {
        for (int i = 0; i < mesh.nx(); ++i) {
            DebugCell* cell = mesh.getTypedCell(i, j);
            bool shouldBeInRegion = cell->value > 50.0;
            bool isInRegion = region.contains(cell);
            
            std::cout << "Cell (" << i << "," << j << ") with value " << cell->value 
                      << ": Should be in region: " << shouldBeInRegion 
                      << ", Is in region: " << isInRegion << std::endl;
            
            if (shouldBeInRegion == isInRegion) {
                successCount++;
            } else {
                failCount++;
            }
            
            EXPECT_EQ(shouldBeInRegion, isInRegion)
                << "Cell at (" << i << "," << j << ") with value " << cell->value 
                << (shouldBeInRegion ? " should" : " should not") << " be in the region";
        }
    }
    
    std::cout << "Success count: " << successCount << ", Fail count: " << failCount << std::endl;
}

