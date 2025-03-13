#include <gtest/gtest.h>
#include <chrono>
#include <iostream>
#include "BitArray.h"

TEST(BitArrayPerformanceTest, CountPerformanceWithDifferentDensities) {
    const size_t size = 1000000;
    
    // Sparse array (1% set)
    BitArray sparseArray(size, false);
    for (size_t i = 0; i < size; i += 100) {
        sparseArray.set(i, true);
    }
    
    // Medium array (50% set)
    BitArray mediumArray(size, false);
    for (size_t i = 0; i < size; i += 2) {
        mediumArray.set(i, true);
    }
    
    // Dense array (99% set)
    BitArray denseArray(size, true);
    for (size_t i = 0; i < size; i += 100) {
        denseArray.set(i, false);
    }
    
    auto startSparse = std::chrono::high_resolution_clock::now();
    size_t sparseCount = sparseArray.count();
    auto endSparse = std::chrono::high_resolution_clock::now();
    
    auto startMedium = std::chrono::high_resolution_clock::now();
    size_t mediumCount = mediumArray.count();
    auto endMedium = std::chrono::high_resolution_clock::now();
    
    auto startDense = std::chrono::high_resolution_clock::now();
    size_t denseCount = denseArray.count();
    auto endDense = std::chrono::high_resolution_clock::now();
    
    auto sparseDuration = std::chrono::duration_cast<std::chrono::microseconds>(endSparse - startSparse).count();
    auto mediumDuration = std::chrono::duration_cast<std::chrono::microseconds>(endMedium - startMedium).count();
    auto denseDuration = std::chrono::duration_cast<std::chrono::microseconds>(endDense - startDense).count();
    
    EXPECT_EQ(size / 100, sparseCount);
    EXPECT_EQ(size / 2, mediumCount);
    EXPECT_EQ(size - size / 100, denseCount);
    
    // Log performance metrics (don't actually assert on timing since it's machine-dependent)
    std::cout << "Count time for sparse array (1%): " << sparseDuration << " µs\n";
    std::cout << "Count time for medium array (50%): " << mediumDuration << " µs\n";
    std::cout << "Count time for dense array (99%): " << denseDuration << " µs\n";
}

TEST(BitArrayPerformanceTest, TraversalPerformanceWithDifferentDensities) {
    const size_t size = 1000000;
    
    // Sparse array (1% set)
    BitArray sparseArray(size, false);
    for (size_t i = 0; i < size; i += 100) {
        sparseArray.set(i, true);
    }
    
    // Medium array (10% set)
    BitArray mediumArray(size, false);
    for (size_t i = 0; i < size; i += 10) {
        mediumArray.set(i, true);
    }
    
    // Measure traversal performance using findFirst/findNext
    auto startSparse = std::chrono::high_resolution_clock::now();
    size_t sparseVisited = 0;
    for (size_t idx = sparseArray.findFirst(); idx < sparseArray.size(); idx = sparseArray.findNext(idx)) {
        sparseVisited++;
    }
    auto endSparse = std::chrono::high_resolution_clock::now();
    
    auto startMedium = std::chrono::high_resolution_clock::now();
    size_t mediumVisited = 0;
    for (size_t idx = mediumArray.findFirst(); idx < mediumArray.size(); idx = mediumArray.findNext(idx)) {
        mediumVisited++;
    }
    auto endMedium = std::chrono::high_resolution_clock::now();
    
    auto sparseDuration = std::chrono::duration_cast<std::chrono::microseconds>(endSparse - startSparse).count();
    auto mediumDuration = std::chrono::duration_cast<std::chrono::microseconds>(endMedium - startMedium).count();
    
    EXPECT_EQ(size / 100, sparseVisited);
    EXPECT_EQ(size / 10, mediumVisited);
    
    // Log performance metrics
    std::cout << "Traversal time for sparse array (1%): " << sparseDuration << " µs, visited " << sparseVisited << " bits\n";
    std::cout << "Traversal time for medium array (10%): " << mediumDuration << " µs, visited " << mediumVisited << " bits\n";
}

TEST(BitArrayPerformanceTest, BitwiseOperationPerformance) {
    const size_t size = 1000000;
    
    BitArray a(size, false);
    BitArray b(size, false);
    
    // Set every 3rd bit in a
    for (size_t i = 0; i < size; i += 3) {
        a.set(i, true);
    }
    
    // Set every 5th bit in b
    for (size_t i = 0; i < size; i += 5) {
        b.set(i, true);
    }
    
    // Measure performance of bitwise operations
    auto startOr = std::chrono::high_resolution_clock::now();
    BitArray orResult(a);
    orResult.bitwiseOr(b);
    auto endOr = std::chrono::high_resolution_clock::now();
    
    auto startAnd = std::chrono::high_resolution_clock::now();
    BitArray andResult(a);
    andResult.bitwiseAnd(b);
    auto endAnd = std::chrono::high_resolution_clock::now();
    
    auto startXor = std::chrono::high_resolution_clock::now();
    BitArray xorResult(a);
    xorResult.bitwiseXor(b);
    auto endXor = std::chrono::high_resolution_clock::now();
    
    auto orDuration = std::chrono::duration_cast<std::chrono::microseconds>(endOr - startOr).count();
    auto andDuration = std::chrono::duration_cast<std::chrono::microseconds>(endAnd - startAnd).count();
    auto xorDuration = std::chrono::duration_cast<std::chrono::microseconds>(endXor - startXor).count();
    
    // Log performance metrics
    std::cout << "OR operation time: " << orDuration << " µs\n";
    std::cout << "AND operation time: " << andDuration << " µs\n";
    std::cout << "XOR operation time: " << xorDuration << " µs\n";
    
    // Verify correctness of operations
    size_t orCount = orResult.count();
    size_t andCount = andResult.count();
    size_t xorCount = xorResult.count();
    
    std::cout << "OR result count: " << orCount << "\n";
    std::cout << "AND result count: " << andCount << "\n";
    std::cout << "XOR result count: " << xorCount << "\n";
}