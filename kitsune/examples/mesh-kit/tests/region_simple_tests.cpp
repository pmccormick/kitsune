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
class TestCell : public CellBase {
public:
    TestCell() : CellBase(0, 0, nullptr), value(0.0) {}
    
    TestCell(int i, int j, MeshBase* mesh) : CellBase(i, j, mesh), value(0.0) {}
    
    double value;
};

// Simple test mesh implementation
class TestMesh : public MeshBase {
public:
    TestMesh(int nx, int ny) : MeshBase(nx, ny, 1.0, 1.0) {
        // Initialize cells
        m_cells.resize(nx * ny);
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                m_cells[linearIndex(i, j)] = TestCell(i, j, this);
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
    
    TestCell* getTypedCell(int i, int j) {
        if (i < 0 || i >= nx() || j < 0 || j >= ny()) {
            return nullptr;
        }
        return &m_cells[linearIndex(i, j)];
    }
    
private:
    std::vector<TestCell> m_cells;
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

TEST(RegionDebugTest, AddCellTest) {
    TestMesh mesh(10, 10);
    
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
        TestCell* cell = mesh.getTypedCell(i, 0);
        bool contains = region.contains(cell);
        std::cout << "Contains cell (" << i << ",0): " << contains << std::endl;
        EXPECT_TRUE(contains) << "Cell at (" << i << ",0) should be in the region";
    }
}


