/**
 * @file CellPerformanceTests.cpp
 * @brief Performance tests for the Cell class
 */

#include <gtest/gtest.h>
#include "Mesh.h"
#include "Cell.h"
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

namespace mesh {
namespace testing {

/**
 * @brief Test fixture for Cell performance tests
 */
class CellPerformanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create mesh for testing
        standardMesh = new Mesh(1000, 1000);
    }

    void TearDown() override {
        delete standardMesh;
    }

    // Helper method to measure execution time of a function
    template<typename Func>
    double measureExecutionTime(Func func, int iterations = 1) {
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < iterations; ++i) {
            func();
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        return duration.count() / iterations;
    }

    // Mesh for testing
    Mesh* standardMesh;
};

/**
 * @brief Test memory footprint of Cell instances
 */
TEST_F(CellPerformanceTest, MemoryFootprint) {
    // Size of Cell should be small (mesh pointer + two indices)
    size_t cellSize = sizeof(Cell);
    
    // Print cell size for information (this is not a pass/fail test)
    std::cout << "Size of Cell: " << cellSize << " bytes" << std::endl;
    
    // Expected size: one pointer (8 bytes on 64-bit systems) + two indices (4 bytes each)
    // but need to account for potential alignment/padding
    EXPECT_LE(cellSize, 32); // Should be <= 32 bytes even with padding
    
    // Size should be significantly smaller than a hypothetical cell storing
    // all its data directly
    struct HypotheticalFullCell {
        Mesh* mesh;
        int i, j;
        double x, y;           // Physical coordinates
        std::vector<double> data; // Some data
    };
    
    EXPECT_LT(cellSize, sizeof(HypotheticalFullCell));
}

/**
 * @brief Test performance of basic Cell operations
 */
TEST_F(CellPerformanceTest, BasicOperationsPerformance) {
    // Create a grid of cells for testing
    const int gridSize = 100;
    std::vector<Cell> cells;
    cells.reserve(gridSize * gridSize);
    
    for (int j = 0; j < gridSize; ++j) {
        for (int i = 0; i < gridSize; ++i) {
            cells.push_back(standardMesh->getCell(i, j));
        }
    }
    
    // Measure accessors performance
    double accessorTime = measureExecutionTime([&cells]() {
        volatile int sum = 0;
        for (const auto& cell : cells) {
            sum += cell.i() + cell.j();
        }
    }, 100);
    
    // Measure linearIndex performance
    double linearIndexTime = measureExecutionTime([&cells]() {
        volatile int sum = 0;
        for (const auto& cell : cells) {
            sum += cell.linearIndex();
        }
    }, 100);
    
    // Measure isBoundary performance
    double isBoundaryTime = measureExecutionTime([&cells]() {
        volatile int count = 0;
        for (const auto& cell : cells) {
            if (cell.isBoundary()) {
                count = count + 1;
            }
        }
    }, 100);
    
    // Measure neighbor finding performance
    double neighborTime = measureExecutionTime([&cells]() {
        volatile int sum = 0;
        for (const auto& cell : cells) {
            Cell neighbor = cell.neighbor(NORTH);
            if (neighbor.isValid()) {
                sum += neighbor.i() + neighbor.j();
            }
        }
    }, 100);
    
    // Print results (not pass/fail tests)
    std::cout << "Average time for accessors: " << accessorTime << " ms" << std::endl;
    std::cout << "Average time for linearIndex: " << linearIndexTime << " ms" << std::endl;
    std::cout << "Average time for isBoundary: " << isBoundaryTime << " ms" << std::endl;
    std::cout << "Average time for neighbor finding: " << neighborTime << " ms" << std::endl;
    
    // No specific pass/fail criteria, but linearIndex and isBoundary should take longer
    // than basic accessors due to more complex operations
    EXPECT_LT(accessorTime, linearIndexTime + isBoundaryTime);
}

/**
 * @brief Test performance of navigating a mesh using Cell
 */
TEST_F(CellPerformanceTest, MeshNavigationPerformance) {
    // Starting cell
    Cell cell = standardMesh->getCell(500, 500);
    
    // Measure performance of spiral navigation
    double spiralTime = measureExecutionTime([&cell]() {
        // Navigate in a spiral pattern
        int steps = 100;
        uint8_t directions[] = {EAST, NORTH, WEST, SOUTH};
        int dirIndex = 0;
        int stepCount = 1;
        int stepsTaken = 0;
        
        volatile int sum = 0;
        
        for (int i = 0; i < steps; ++i) {
            Cell neighbor = cell.neighbor(directions[dirIndex]);
            if (neighbor.isValid()) {
                cell = neighbor;
                sum += cell.i() + cell.j();
            }
            
            stepsTaken++;
            if (stepsTaken == stepCount) {
                dirIndex = (dirIndex + 1) % 4;
                stepsTaken = 0;
                
                // Every two direction changes, increase the number of steps
                if (dirIndex % 2 == 0) {
                    stepCount++;
                }
            }
        }
    }, 100);
    
    // Measure performance of random navigation
    double randomTime = measureExecutionTime([&cell]() {
        // Navigate in a random pattern
        int steps = 100;
        std::mt19937 rng(42); // Fixed seed for reproducibility
        std::uniform_int_distribution<int> dist(0, 3);
        uint8_t directions[] = {NORTH, EAST, SOUTH, WEST};
        
        volatile int sum = 0;
        
        for (int i = 0; i < steps; ++i) {
            uint8_t dir = directions[dist(rng)];
            Cell neighbor = cell.neighbor(dir);
            if (neighbor.isValid()) {
                cell = neighbor;
                sum += cell.i() + cell.j();
            }
        }
    }, 100);
    
    // Print results
    std::cout << "Average time for spiral navigation: " << spiralTime << " ms" << std::endl;
    std::cout << "Average time for random navigation: " << randomTime << " ms" << std::endl;
    
    // Performance tests are system-dependent, so we only check 
    // that both methods complete in a reasonable time
    EXPECT_LT(spiralTime, 10.0); // Less than 10ms
    EXPECT_LT(randomTime, 10.0); // Less than 10ms
}

/**
 * @brief Test performance of bulk operations on cells
 */
TEST_F(CellPerformanceTest, BulkOperationsPerformance) {
    // Create a large number of cells
    const int numCells = 10000;
    std::vector<Cell> cells;
    cells.reserve(numCells);
    
    // Random positions
    std::mt19937 rng(42); // Fixed seed for reproducibility
    std::uniform_int_distribution<int> dist(0, 999);
    
    for (int i = 0; i < numCells; ++i) {
        cells.push_back(standardMesh->getCell(dist(rng), dist(rng)));
    }
    
    // Measure sorting performance
    double sortTime = measureExecutionTime([&cells]() {
        // Sort cells by linearIndex
        std::sort(cells.begin(), cells.end(), [](const Cell& a, const Cell& b) {
            return a.linearIndex() < b.linearIndex();
        });
    }, 10);
    
    // Measure filtering performance
    double filterTime = measureExecutionTime([&cells]() {
        // Filter boundary cells
        std::vector<Cell> boundaryOnly;
        boundaryOnly.reserve(cells.size() / 10); // Approximately 10% will be boundaries
        
        for (const auto& cell : cells) {
            if (cell.isBoundary()) {
                boundaryOnly.push_back(cell);
            }
        }
    }, 10);
    
    // Print results
    std::cout << "Average time for sorting cells: " << sortTime << " ms" << std::endl;
    std::cout << "Average time for filtering boundary cells: " << filterTime << " ms" << std::endl;
    
    // Sorting should be slower than filtering since it involves
    // more comparisons and data movement
    EXPECT_GT(sortTime, filterTime);
}

/**
 * @brief Test performance impact of inlining
 */
TEST_F(CellPerformanceTest, InliningPerformance) {
    // Create test cells
    std::vector<Cell> cells;
    cells.reserve(1000);
    
    for (int i = 0; i < 1000; ++i) {
        cells.push_back(standardMesh->getCell(i % 100, i / 100));
    }
    
    // Simple loop with inline-friendly code
    double inlineTime = measureExecutionTime([&cells]() {
        volatile int sum = 0;
        for (const auto& cell : cells) {
            // These operations should benefit from inlining
            sum += cell.i() + cell.j();
        }
    }, 1000);
    
    // More complex loop that is harder to inline effectively
    double complexTime = measureExecutionTime([&cells]() {
        volatile int sum = 0;
        for (const auto& cell : cells) {
            // These operations are more complex and may not inline as effectively
            if (cell.isValid() && !cell.isBoundary()) {
                Cell n = cell.neighbor(NORTH);
                if (n.isValid()) {
                    sum += n.linearIndex();
                }
            }
        }
    }, 1000);
    
    // Print results
    std::cout << "Average time for inline-friendly operations: " << inlineTime << " ms" << std::endl;
    std::cout << "Average time for complex operations: " << complexTime << " ms" << std::endl;
    
    // Complex operations should take significantly longer
    EXPECT_LT(inlineTime, complexTime);
}

/**
 * @brief Test cache performance with different traversal patterns
 */
TEST_F(CellPerformanceTest, CachePerformance) {
    const int size = 256;
    std::vector<Cell> cells;
    cells.reserve(size * size);
    
    for (int j = 0; j < size; ++j) {
        for (int i = 0; i < size; ++i) {
            cells.push_back(standardMesh->getCell(i, j));
        }
    }
    
    // Row-major traversal (cache-friendly)
    double rowMajorTime = measureExecutionTime([&cells]() {
        const int size = 256; // Move inside lambda
        volatile int sum = 0;
        for (int j = 0; j < size; ++j) {
            for (int i = 0; i < size; ++i) {
                const Cell& cell = cells[j * size + i];
                sum += cell.linearIndex();
            }
        }
    }, 10);
    
    // Column-major traversal (less cache-friendly)
    double colMajorTime = measureExecutionTime([&cells]() {
        const int size = 256; // Move inside lambda
        volatile int sum = 0;
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                const Cell& cell = cells[j * size + i];
                sum += cell.linearIndex();
            }
        }
    }, 10);
    
    // Print results
    std::cout << "Average time for row-major traversal: " << rowMajorTime << " ms" << std::endl;
    std::cout << "Average time for column-major traversal: " << colMajorTime << " ms" << std::endl;
    
    // Row-major should be faster due to better cache locality
    EXPECT_LT(rowMajorTime, colMajorTime);
}

} // namespace testing
} // namespace mesh


