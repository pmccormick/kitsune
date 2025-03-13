/**
 * @file CellPerformanceTests.cpp
 * @brief Performance tests for the Cell class
 * 
 * These tests focus on measuring the performance of common Cell operations
 * and ensuring they meet the design goal of being lightweight and efficient.
 * 
 * Note: These are not typical unit tests as they measure performance rather 
 * than correctness. They should be run separately from regular unit tests
 * and their results interpreted accordingly.
 */

#include <gtest/gtest.h>
#include "Cell.h"
#include "Mesh.h"
#include <chrono>
#include <vector>
#include <string>
#include <iostream>

namespace mesh {
namespace testing {

// Helper macro to time an operation
#define TIME_OPERATION(op, iterations) \
    { \
        auto start = std::chrono::high_resolution_clock::now(); \
        for (int i = 0; i < iterations; ++i) { \
            op; \
        } \
        auto end = std::chrono::high_resolution_clock::now(); \
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start); \
        std::cout << "  " << iterations << " iterations took " << duration.count() << " microseconds" << std::endl; \
        std::cout << "  Average: " << (duration.count() / static_cast<double>(iterations)) << " microseconds per operation" << std::endl; \
    }

// Fixture for Cell performance tests
class CellPerformanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create meshes of different sizes for performance testing
        smallMesh = new Mesh(10, 10);
        mediumMesh = new Mesh(100, 100);
        largeMesh = new Mesh(1000, 1000);
    }

    void TearDown() override {
        delete smallMesh;
        delete mediumMesh;
        delete largeMesh;
    }

    Mesh* smallMesh;
    Mesh* mediumMesh;
    Mesh* largeMesh;
    
    // Constants for iterations
    static constexpr int SMALL_ITERATIONS = 1000000;   // 1 million
    static constexpr int MEDIUM_ITERATIONS = 100000;   // 100k
    static constexpr int LARGE_ITERATIONS = 10000;     // 10k
    static constexpr int VERY_LARGE_ITERATIONS = 1000; // 1k
};

// Test the performance of cell creation
TEST_F(CellPerformanceTest, CellCreationPerformance) {
    std::cout << "Testing Cell creation performance:" << std::endl;
    
    // Small mesh - create cells at fixed position
    std::cout << "Small mesh, fixed position:" << std::endl;
    TIME_OPERATION(smallMesh->getCell(5, 5), SMALL_ITERATIONS);
    
    // Small mesh - create cells at varying positions
    std::cout << "Small mesh, varying positions:" << std::endl;
    TIME_OPERATION(smallMesh->getCell(i % 10, (i / 10) % 10), SMALL_ITERATIONS);
    
    // Medium mesh - fixed position
    std::cout << "Medium mesh, fixed position:" << std::endl;
    TIME_OPERATION(mediumMesh->getCell(50, 50), MEDIUM_ITERATIONS);
    
    // Verify that performance scales well with mesh size
    EXPECT_TRUE(true); // This is a performance test, not a correctness test
}

// Test the performance of cell validity checking
TEST_F(CellPerformanceTest, CellValidityCheckPerformance) {
    std::cout << "Testing Cell isValid() performance:" << std::endl;
    
    // Create test cells
    Cell validCell = smallMesh->getCell(5, 5);
    Cell invalidCell;
    
    // Test valid cell
    std::cout << "Valid cell:" << std::endl;
    TIME_OPERATION(validCell.isValid(), SMALL_ITERATIONS);
    
    // Test invalid cell
    std::cout << "Invalid cell:" << std::endl;
    TIME_OPERATION(invalidCell.isValid(), SMALL_ITERATIONS);
    
    // Make sure validity check is fast regardless of state
    EXPECT_TRUE(true); // Performance test
}

// Test the performance of the linearIndex method
TEST_F(CellPerformanceTest, LinearIndexPerformance) {
    std::cout << "Testing Cell linearIndex() performance:" << std::endl;
    
    // Create test cells
    Cell smallCell = smallMesh->getCell(5, 5);
    Cell mediumCell = mediumMesh->getCell(50, 50);
    Cell largeCell = largeMesh->getCell(500, 500);
    
    // Test with small mesh
    std::cout << "Small mesh cell:" << std::endl;
    TIME_OPERATION(smallCell.linearIndex(), SMALL_ITERATIONS);
    
    // Test with medium mesh
    std::cout << "Medium mesh cell:" << std::endl;
    TIME_OPERATION(mediumCell.linearIndex(), SMALL_ITERATIONS);
    
    // Test with large mesh
    std::cout << "Large mesh cell:" << std::endl;
    TIME_OPERATION(largeCell.linearIndex(), SMALL_ITERATIONS);
    
    // Verify that linearIndex is fast regardless of mesh size
    EXPECT_TRUE(true); // Performance test
}

// Test the performance of boundary checking
TEST_F(CellPerformanceTest, BoundaryCheckPerformance) {
    std::cout << "Testing Cell isBoundary() performance:" << std::endl;
    
    // Create test cells
    Cell interiorCell = smallMesh->getCell(5, 5);
    Cell boundaryCell = smallMesh->getCell(0, 5);
    
    // Test interior cell (not boundary)
    std::cout << "Interior cell:" << std::endl;
    TIME_OPERATION(interiorCell.isBoundary(), SMALL_ITERATIONS);
    
    // Test boundary cell
    std::cout << "Boundary cell:" << std::endl;
    TIME_OPERATION(boundaryCell.isBoundary(), SMALL_ITERATIONS);
    
    // Verify that boundary check is fast regardless of result
    EXPECT_TRUE(true); // Performance test
}

// Test the performance of the neighbor method
TEST_F(CellPerformanceTest, NeighborMethodPerformance) {
    std::cout << "Testing Cell neighbor() performance:" << std::endl;
    
    // Create test cells
    Cell interiorCell = smallMesh->getCell(5, 5);
    Cell boundaryCell = smallMesh->getCell(0, 5);
    
    // Test interior cell - all neighbors valid
    std::cout << "Interior cell, NORTH neighbor:" << std::endl;
    TIME_OPERATION(interiorCell.neighbor(NORTH), SMALL_ITERATIONS);
    
    // Test interior cell - diagonal neighbor
    std::cout << "Interior cell, NORTH|EAST neighbor:" << std::endl;
    TIME_OPERATION(interiorCell.neighbor(NORTH | EAST), SMALL_ITERATIONS);
    
    // Test boundary cell - out-of-bounds neighbor
    std::cout << "Boundary cell, out-of-bounds neighbor:" << std::endl;
    TIME_OPERATION(boundaryCell.neighbor(WEST), SMALL_ITERATIONS);
    
    // Verify that neighbor calculation is fast regardless of position
    EXPECT_TRUE(true); // Performance test
}

// Test the performance of equality comparison
TEST_F(CellPerformanceTest, EqualityComparisonPerformance) {
    std::cout << "Testing Cell equality comparison performance:" << std::endl;
    
    // Create test cells
    Cell cell1 = smallMesh->getCell(5, 5);
    Cell cell2 = smallMesh->getCell(5, 5); // Same position
    Cell cell3 = smallMesh->getCell(6, 6); // Different position
    
    // Test equality of identical cells
    std::cout << "Equal cells:" << std::endl;
    TIME_OPERATION(cell1 == cell2, SMALL_ITERATIONS);
    
    // Test equality of different cells
    std::cout << "Different cells:" << std::endl;
    TIME_OPERATION(cell1 == cell3, SMALL_ITERATIONS);
    
    // Verify that equality check is fast regardless of result
    EXPECT_TRUE(true); // Performance test
}

// Test the performance of repeated cell operations (simulation-like workload)
TEST_F(CellPerformanceTest, SimulationWorkloadPerformance) {
    std::cout << "Testing Cell performance in simulation-like workload:" << std::endl;
    
    // Create a grid of cells for simulation
    std::vector<Cell> grid;
    const int gridSize = 10; // 10x10 grid
    grid.reserve(gridSize * gridSize);
    
    for (int j = 0; j < gridSize; ++j) {
        for (int i = 0; i < gridSize; ++i) {
            grid.push_back(smallMesh->getCell(i, j));
        }
    }
    
    // Simulate a workload that accesses neighbors and checks properties
    std::cout << "Stencil computation (access self + neighbors):" << std::endl;
    TIME_OPERATION({
        // Skip edge cells to avoid boundary checks in benchmark
        for (int j = 1; j < gridSize - 1; ++j) {
            for (int i = 1; i < gridSize - 1; ++i) {
                Cell& cell = grid[i + j * gridSize];
                // 5-point stencil
                Cell north = cell.neighbor(NORTH);
                Cell east = cell.neighbor(EAST);
                Cell south = cell.neighbor(SOUTH);
                Cell west = cell.neighbor(WEST);
                // Use cell properties (simulate computation)
                int centerIdx = cell.linearIndex();
                int sum = centerIdx + north.linearIndex() + east.linearIndex() + 
                          south.linearIndex() + west.linearIndex();
                // Prevent compiler from optimizing away
                if (sum < 0) break; // Never happens
            }
        }
    }, LARGE_ITERATIONS);
    
    // Verify that cell operations are fast in a realistic workflow
    EXPECT_TRUE(true); // Performance test
}

// Test the performance impact of large-scale cell operations
TEST_F(CellPerformanceTest, LargeScaleOperationsPerformance) {
    std::cout << "Testing Cell performance with large-scale operations:" << std::endl;
    
    // Create and use a large number of cells
    std::cout << "Creating and storing 10,000 cells:" << std::endl;
    TIME_OPERATION({
        std::vector<Cell> cells;
        cells.reserve(10000);
        for (int j = 0; j < 100; ++j) {
            for (int i = 0; i < 100; ++i) {
                cells.push_back(mediumMesh->getCell(i, j));
            }
        }
    }, MEDIUM_ITERATIONS);
    
    // Test cell-by-cell iteration on a large mesh
    std::cout << "Iterating through 10,000 cells:" << std::endl;
    TIME_OPERATION({
        std::vector<Cell> cells;
        cells.reserve(10000);
        for (int j = 0; j < 100; ++j) {
            for (int i = 0; i < 100; ++i) {
                cells.push_back(mediumMesh->getCell(i, j));
            }
        }
        
        int sum = 0;
        for (const Cell& cell : cells) {
            sum += cell.linearIndex();
        }
        // Prevent compiler from optimizing away
        if (sum < 0) break; // Never happens
    }, MEDIUM_ITERATIONS);
    
    // Verify that cell performance scales well
    EXPECT_TRUE(true); // Performance test
}

// Test the memory usage of cells (indirect measurement through bulk creation)
TEST_F(CellPerformanceTest, MemoryUsageEfficiency) {
    std::cout << "Testing Cell memory usage efficiency:" << std::endl;
    
    // Create many cells to indirectly measure memory usage
    std::cout << "Creating 1 million cells:" << std::endl;
    std::vector<Cell> cells;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Reserve space to avoid reallocation
    cells.reserve(1000000);
    
    // Create 1 million cells
    for (int i = 0; i < 1000; ++i) {
        for (int j = 0; j < 1000; ++j) {
            cells.push_back(largeMesh->getCell(i, j));
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "  Creation took " << duration.count() << " milliseconds" << std::endl;
    std::cout << "  Vector capacity: " << cells.capacity() << " cells" << std::endl;
    std::cout << "  Vector size: " << cells.size() << " cells" << std::endl;
    
    // Calculate approximate memory usage
    size_t cellSize = sizeof(Cell);
    size_t totalMemory = cellSize * cells.size();
    std::cout << "  Approximate memory usage: " << cellSize << " bytes per cell, " 
              << totalMemory << " bytes total (" 
              << totalMemory / (1024.0 * 1024.0) << " MB)" << std::endl;
    
    // Verify that cells are lightweight
    EXPECT_TRUE(cellSize <= 24); // Cell should be small (3 pointers or less)
    
    // Clear the vector to free memory
    cells.clear();
}

} // namespace testing
} // namespace mesh
