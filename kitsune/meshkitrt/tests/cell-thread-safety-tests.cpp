/**
 * @file CellThreadSafetyTests.cpp
 * @brief Unit tests for thread safety of the Cell class
 */

#include <gtest/gtest.h>
#include "Mesh.h"
#include "Cell.h"
#include <thread>
#include <vector>
#include <atomic>
#include <mutex>

namespace mesh {
namespace testing {

/**
 * @brief Test fixture for Cell thread safety tests
 */
class CellThreadSafetyTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create standard mesh for testing
        standardMesh = new Mesh(100, 100);
    }

    void TearDown() override {
        delete standardMesh;
    }

    // Mesh for testing
    Mesh* standardMesh;
};

/**
 * @brief Test concurrent reads from multiple threads
 */
TEST_F(CellThreadSafetyTest, ConcurrentReads) {
    // Create a set of cells to test
    std::vector<Cell> cells;
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            cells.push_back(standardMesh->getCell(i, j));
        }
    }

    // Atomic counter to track any issues
    std::atomic<int> errorCount(0);
    
    // Create threads to concurrently read from cells
    std::vector<std::thread> threads;
    const int NUM_THREADS = 10;
    
    for (int t = 0; t < NUM_THREADS; t++) {
        threads.push_back(std::thread([&cells, &errorCount]() {
            // Each thread reads from all cells multiple times
            for (int iter = 0; iter < 100; iter++) {
                for (const Cell& cell : cells) {
                    try {
                        // Call various const methods
                        int i = cell.i();
                        int j = cell.j();
                        auto indices = cell.indices();
                        (void)cell.mesh();
                        (void)cell.linearIndex();
                        (void)cell.isBoundary();
                        (void)cell.isValid();
                        
                        // Verify consistency of values
                        if (i != indices.first || j != indices.second) {
                            errorCount++;
                        }
                    } catch (const std::exception& e) {
                        // Unexpected exception
                        errorCount++;
                    }
                }
            }
        }));
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Verify no errors occurred
    EXPECT_EQ(errorCount.load(), 0);
}

/**
 * @brief Test that const methods don't modify state
 */
TEST_F(CellThreadSafetyTest, ConstMethodsNoModification) {
    // Create a cell
    Cell cell = standardMesh->getCell(5, 5);
    
    // Store original values for comparison
    int originalI = cell.i();
    int originalJ = cell.j();
    Mesh* originalMesh = cell.mesh();
    bool originalValidity = cell.isValid();
    
    // Call a series of const methods that should not modify state
    for (int i = 0; i < 1000; i++) {
        cell.i();
        cell.j();
        cell.indices();
        cell.mesh();
        cell.linearIndex();
        cell.isBoundary();
        cell.isValid();
        cell.neighbor(NORTH);
        cell.neighbor(EAST);
        cell.neighbor(SOUTH);
        cell.neighbor(WEST);
    }
    
    // Verify state hasn't changed
    EXPECT_EQ(cell.i(), originalI);
    EXPECT_EQ(cell.j(), originalJ);
    EXPECT_EQ(cell.mesh(), originalMesh);
    EXPECT_EQ(cell.isValid(), originalValidity);
}

/**
 * @brief Test that parallel neighbor operations are thread-safe
 */
TEST_F(CellThreadSafetyTest, ParallelNeighborOperations) {
    // Create a central cell for testing
    Cell cell = standardMesh->getCell(50, 50);
    
    // Atomic counter to track any issues
    std::atomic<int> errorCount(0);
    
    // Create threads to perform neighbor operations
    std::vector<std::thread> threads;
    const int NUM_THREADS = 8;
    
    for (int t = 0; t < NUM_THREADS; t++) {
        threads.push_back(std::thread([&cell, &errorCount, t]() {
            // Each thread performs a different direction of neighbor operation
            uint8_t direction = 1 << (t % 4); // NORTH, EAST, SOUTH, or WEST
            
            for (int iter = 0; iter < 1000; iter++) {
                try {
                    Cell neighbor = cell.neighbor(direction);
                    
                    // Verify neighbor relationship
                    if (direction == NORTH && neighbor.j() != cell.j() + 1) {
                        errorCount++;
                    } else if (direction == EAST && neighbor.i() != cell.i() + 1) {
                        errorCount++;
                    } else if (direction == SOUTH && neighbor.j() != cell.j() - 1) {
                        errorCount++;
                    } else if (direction == WEST && neighbor.i() != cell.i() - 1) {
                        errorCount++;
                    }
                } catch (const std::exception& e) {
                    // Unexpected exception
                    errorCount++;
                }
            }
        }));
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Verify no errors occurred
    EXPECT_EQ(errorCount.load(), 0);
}

/**
 * @brief Test equality operators in multi-threaded context
 */
TEST_F(CellThreadSafetyTest, ConcurrentEqualityComparisons) {
    // Create cells for comparison
    Cell cell1 = standardMesh->getCell(25, 25);
    Cell cell2 = standardMesh->getCell(25, 25); // Same position
    Cell cell3 = standardMesh->getCell(75, 75); // Different position
    
    // Atomic counter to track any issues
    std::atomic<int> errorCount(0);
    
    // Create threads for concurrent comparisons
    std::vector<std::thread> threads;
    const int NUM_THREADS = 10;
    
    for (int t = 0; t < NUM_THREADS; t++) {
        threads.push_back(std::thread([&cell1, &cell2, &cell3, &errorCount]() {
            for (int iter = 0; iter < 1000; iter++) {
                // Same position cells should be equal
                if (!(cell1 == cell2) || (cell1 != cell2)) {
                    errorCount++;
                }
                
                // Different position cells should not be equal
                if ((cell1 == cell3) || !(cell1 != cell3)) {
                    errorCount++;
                }
            }
        }));
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Verify no errors occurred
    EXPECT_EQ(errorCount.load(), 0);
}

/**
 * @brief Test reading cells in different threads after mesh modifications
 * 
 * This is a more complex test that verifies cell views remain valid
 * even when other threads are working with the mesh
 */
TEST_F(CellThreadSafetyTest, CellViewConsistency) {
    // Vector to hold cells
    std::vector<Cell> cells;
    std::mutex cellsMutex;
    
    // Create thread to continuously create cells
    std::atomic<bool> done(false);
    std::thread cellCreator([this, &cells, &cellsMutex, &done]() {
        for (int i = 0; i < 10 && !done; i++) {
            for (int j = 0; j < 10 && !done; j++) {
                Cell cell = standardMesh->getCell(i, j);
                
                // Store cell in vector
                {
                    std::lock_guard<std::mutex> lock(cellsMutex);
                    cells.push_back(cell);
                }
                
                // Small delay to allow reader threads to work
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    });
    
    // Create threads to read from cells
    std::atomic<int> errorCount(0);
    std::vector<std::thread> readers;
    const int NUM_READERS = 5;
    
    for (int r = 0; r < NUM_READERS; r++) {
        readers.push_back(std::thread([&cells, &cellsMutex, &errorCount, &done]() {
            while (!done) {
                // Copy cells to local vector to avoid holding lock during processing
                std::vector<Cell> localCells;
                {
                    std::lock_guard<std::mutex> lock(cellsMutex);
                    localCells = cells;
                }
                
                // Skip if no cells yet
                if (localCells.empty()) {
                    std::this_thread::yield();
                    continue;
                }
                
                // Process cells
                for (const Cell& cell : localCells) {
                    try {
                        // Verify cell consistency
                        int i = cell.i();
                        int j = cell.j();
                        auto indices = cell.indices();
                        
                        if (i != indices.first || j != indices.second) {
                            errorCount++;
                        }
                    } catch (const std::exception& e) {
                        // Unexpected exception
                        errorCount++;
                    }
                }
                
                // Small delay to reduce CPU usage
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }));
    }
    
    // Let threads run for a short time
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    done = true;
    
    // Join all threads
    cellCreator.join();
    for (auto& thread : readers) {
        thread.join();
    }
    
    // Verify no errors occurred
    EXPECT_EQ(errorCount.load(), 0);
}

/**
 * @brief Test that cells remain valid across threads after creation
 */
TEST_F(CellThreadSafetyTest, CellValidityAcrossThreads) {
    // Create cells in the main thread
    std::vector<Cell> cells;
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            cells.push_back(standardMesh->getCell(i, j));
        }
    }
    
    // Atomic counter to track any issues
    std::atomic<int> errorCount(0);
    
    // Create a thread to verify the cells
    std::thread verifier([&cells, &errorCount]() {
        for (const Cell& cell : cells) {
            // Every cell should be valid
            if (!cell.isValid()) {
                errorCount++;
            }
            
            // Original indices should be preserved
            int expectedI = cell.i();
            int expectedJ = cell.j();
            
            // Check repeatedly
            for (int iter = 0; iter < 100; iter++) {
                if (cell.i() != expectedI || cell.j() != expectedJ) {
                    errorCount++;
                }
            }
        }
    });
    
    // Wait for verification to complete
    verifier.join();
    
    // Verify no errors occurred
    EXPECT_EQ(errorCount.load(), 0);
}

} // namespace testing
} // namespace mesh


