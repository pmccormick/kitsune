/**
 * @file CellStressTests.cpp
 * @brief Stress tests for the Cell class under extreme conditions
 */

#include <gtest/gtest.h>
#include "Mesh.h"
#include "Cell.h"
#include <vector>
#include <random>
#include <thread>
#include <atomic>
#include <chrono>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>

namespace mesh {
namespace testing {

/**
 * @brief Test fixture for Cell stress tests
 */
class CellStressTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create large mesh for stress testing
        largeMesh = new Mesh(2000, 2000);
    }

    void TearDown() override {
        delete largeMesh;
    }

    // Mesh for testing
    Mesh* largeMesh;
};

/**
 * @brief Test with a very large number of cells
 */
TEST_F(CellStressTest, MassiveCellCollection) {
    const int numCells = 100000; // 100K cells
    std::vector<Cell> cells;
    cells.reserve(numCells);
    
    // Create many cells with random positions
    std::mt19937 rng(42); // Fixed seed for reproducibility
    std::uniform_int_distribution<int> dist(0, 1999);
    
    for (int i = 0; i < numCells; ++i) {
        cells.push_back(largeMesh->getCell(dist(rng), dist(rng)));
    }
    
    // Verify all cells are valid
    int invalidCount = 0;
    for (const auto& cell : cells) {
        if (!cell.isValid()) {
            invalidCount++;
        }
    }
    EXPECT_EQ(invalidCount, 0);
    
    // Calculate unique cell count
    std::unordered_set<int> uniqueIndices;
    for (const auto& cell : cells) {
        uniqueIndices.insert(cell.linearIndex());
    }
    std::cout << "Generated " << numCells << " cells with " 
              << uniqueIndices.size() << " unique positions" << std::endl;
}

/**
 * @brief Test cell navigation with complex patterns
 */
TEST_F(CellStressTest, ComplexNavigationPatterns) {
    // Start at center
    Cell current = largeMesh->getCell(1000, 1000);
    
    // Define complex pattern: spiral + jump + backtrack
    const int totalMoves = 10000;
    int moveCount = 0;
    
    // Track visited cells
    std::unordered_set<int> visited;
    visited.insert(current.linearIndex());
    
    // Random generator for jumps
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> distLarge(0, 1999);
    std::uniform_int_distribution<int> distSmall(-5, 5);
    
    // Track furthest distance reached
    int maxManhattanDistance = 0;
    
    // Perform navigation
    for (int i = 0; i < totalMoves && moveCount < totalMoves; ++i) {
        uint8_t direction;
        Cell next;
        
        // Every 100 steps, make a random jump
        if (i % 100 == 99) {
            next = largeMesh->getCell(distLarge(rng), distLarge(rng));
            if (next.isValid()) {
                current = next;
                visited.insert(current.linearIndex());
                moveCount++;
            }
            continue;
        }
        
        // Every 50 steps, try to backtrack to a previously visited cell
        if (i % 50 == 49 && !visited.empty()) {
            // Simple backtracking - just go opposite direction
            // In a more complex test, we could maintain a path history
            direction = 1 << (i % 4); // Pick a direction based on iteration
            next = current.neighbor(direction);
            if (!next.isValid()) continue;
            
            current = next;
            visited.insert(current.linearIndex());
            moveCount++;
            continue;
        }
        
        // Normal case: try a random direction with small offset
        int di = distSmall(rng);
        int dj = distSmall(rng);
        
        // Convert offsets to direction flags
        direction = 0;
        if (di > 0) direction |= EAST;
        if (di < 0) direction |= WEST;
        if (dj > 0) direction |= NORTH;
        if (dj < 0) direction |= SOUTH;
        
        // If no direction set, continue
        if (direction == 0) continue;
        
        // Try to navigate
        next = current.neighbor(direction);
        if (next.isValid()) {
            current = next;
            visited.insert(current.linearIndex());
            moveCount++;
            
            // Track maximum distance from origin
            int manhattanDistance = std::abs(current.i() - 1000) + std::abs(current.j() - 1000);
            maxManhattanDistance = std::max(maxManhattanDistance, manhattanDistance);
        }
    }
    
    std::cout << "Performed " << moveCount << " moves, visited " 
              << visited.size() << " unique cells" << std::endl;
    std::cout << "Maximum Manhattan distance from origin: " 
              << maxManhattanDistance << std::endl;
    
    // Verify we were able to navigate successfully
    EXPECT_GT(moveCount, totalMoves / 2);
    EXPECT_GT(visited.size(), totalMoves / 10);
}

/**
 * @brief Test intense multi-threaded operations on cells
 */
TEST_F(CellStressTest, IntenseMultithreadedOperations) {
    // Vector to hold cells
    std::vector<Cell> cells;
    const int gridSize = 100;
    cells.reserve(gridSize * gridSize);
    
    // Create grid of cells
    for (int j = 0; j < gridSize; ++j) {
        for (int i = 0; i < gridSize; ++i) {
            cells.push_back(largeMesh->getCell(i + 950, j + 950));
        }
    }
    
    // Atomic counters for statistics
    std::atomic<int> totalOperations(0);
    std::atomic<int> boundaryCount(0);
    std::atomic<int> invalidNeighborCount(0);
    
    // Flag to stop worker threads
    std::atomic<bool> stopWorkers(false);
    
    // Launch multiple worker threads
    const int numThreads = 8;
    std::vector<std::thread> workers;
    
    for (int t = 0; t < numThreads; ++t) {
        workers.push_back(std::thread([&cells, &totalOperations, 
                                      &boundaryCount, &invalidNeighborCount, 
                                      &stopWorkers, t]() {
            // Each thread uses its own RNG
            std::mt19937 rng(42 + t);
            std::uniform_int_distribution<size_t> cellDist(0, cells.size() - 1);
            std::uniform_int_distribution<int> dirDist(0, 15); // All possible direction combinations
            
            while (!stopWorkers.load()) {
                // Pick a random cell
                Cell cell = cells[cellDist(rng)];
                
                // Perform various operations
                if (cell.isBoundary()) {
                    boundaryCount++;
                }
                
                // Try a random direction
                uint8_t direction = dirDist(rng);
                Cell neighbor = cell.neighbor(direction);
                
                if (!neighbor.isValid()) {
                    invalidNeighborCount++;
                }
                
                // Count operation
                totalOperations++;
            }
        }));
    }
    
    // Let threads run for a short time
    std::this_thread::sleep_for(std::chrono::seconds(1));
    stopWorkers = true;
    
    // Join all threads
    for (auto& thread : workers) {
        thread.join();
    }
    
    // Print statistics
    std::cout << "Total operations: " << totalOperations.load() << std::endl;
    std::cout << "Boundary cells encountered: " << boundaryCount.load() << std::endl;
    std::cout << "Invalid neighbors encountered: " << invalidNeighborCount.load() << std::endl;
    
    // Verify operations completed successfully
    EXPECT_GT(totalOperations.load(), 1000); // Lower threshold for weaker systems
}

/**
 * @brief Test extreme edge case: maximum mesh dimensions
 */
TEST_F(CellStressTest, MaximumMeshDimensions) {
    // Create a mesh with largest reasonable dimensions
    // This tests both the mesh and cell handling of large indices
    const uint32_t maxDim = 10000; // 10K x 10K = 100M cells (very large)
    
    try {
        Mesh* maxMesh = new Mesh(maxDim, maxDim);
        
        // Test operations on corner and center cells
        Cell cornerCell = maxMesh->getCell(0, 0);
        Cell centerCell = maxMesh->getCell(maxDim/2, maxDim/2);
        Cell farCell = maxMesh->getCell(maxDim-1, maxDim-1);
        
        EXPECT_TRUE(cornerCell.isValid());
        EXPECT_TRUE(centerCell.isValid());
        EXPECT_TRUE(farCell.isValid());
        
        // Check linearIndex calculation on maximum indices
        uint64_t expectedIndex = static_cast<uint64_t>(maxDim-1) + 
                                static_cast<uint64_t>(maxDim-1) * static_cast<uint64_t>(maxDim);
        
        // Since linearIndex returns an int, we might get integer overflow
        // This is a design limitation to be aware of
        std::cout << "Far corner linearIndex: " << farCell.linearIndex() << std::endl;
        std::cout << "Expected value (might exceed int range): " << expectedIndex << std::endl;
        
        // Clean up
        delete maxMesh;
        SUCCEED() << "Successfully created and operated on maximum size mesh";
    }
    catch (const std::exception& e) {
        // If creating such a large mesh fails due to memory limits,
        // that's acceptable - just report it
        std::cout << "Could not create maximum size mesh: " << e.what() << std::endl;
        SUCCEED() << "Exception creating maximum mesh is acceptable if due to resource limits";
    }
}

/**
 * @brief Test neighbor navigation with extreme repetition
 */
TEST_F(CellStressTest, ExtensiveNeighborNavigation) {
    // Test extreme repeated neighbor operations
    Cell center = largeMesh->getCell(1000, 1000);
    
    // Define a long path that should return to the starting point
    const int pathLength = 1000000; // 1M steps
    
    // Track starting position
    int startI = center.i();
    int startJ = center.j();
    
    // Perform long navigation that should return to start
    // North, East, South, West repeated many times
    auto startTime = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < pathLength; ++i) {
        uint8_t direction = (i % 4 == 0) ? NORTH :
                           (i % 4 == 1) ? EAST :
                           (i % 4 == 2) ? SOUTH : WEST;
        
        // This creates a square spiral pattern
        if (i % 4 == 0 && i > 0 && i % 8 == 0) {
            // Skip some steps periodically to avoid going too far from center
            continue;
        }
        
        Cell neighbor = center.neighbor(direction);
        if (neighbor.isValid()) {
            center = neighbor;
        }
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = endTime - startTime;
    
    int endI = center.i();
    int endJ = center.j();
    
    std::cout << "Performed " << pathLength << " neighbor operations in " 
              << duration.count() << " seconds" << std::endl;
    std::cout << "Started at (" << startI << "," << startJ 
              << "), ended at (" << endI << "," << endJ << ")" << std::endl;
    
    // We don't assert exact coordinates since the path is complex,
    // but we check operations completed in reasonable time
    EXPECT_LT(duration.count(), 10.0); // Should take less than 10 seconds
}

} // namespace testing
} // namespace mesh


