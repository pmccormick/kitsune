#include "gtest/gtest.h"
#include "Field.h"
#include "Mesh.h"
#include "FieldStorage.h"
#include <vector>
#include <chrono>
#include <algorithm>
#include <random>
#include <numeric>

// Helper class to measure execution time
class Timer {
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}
    
    double elapsed() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(now - start_).count();
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

// Test fixture for Field performance tests
class FieldPerformanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a large mesh for performance testing
        mesh = new mesh::Mesh(256, 256);
    }

    void TearDown() override {
        delete mesh;
    }

    mesh::Mesh* mesh;
};

// Test 1: Performance of sequential vs random access
TEST_F(FieldPerformanceTest, SequentialVsRandomAccess) {
    const size_t numAccesses = 1000000; // Number of accesses to perform
    mesh::Field<double> field(*mesh, 0.0);
    
    // Generate random indices
    std::vector<std::pair<uint32_t, uint32_t>> randomIndices;
    randomIndices.reserve(numAccesses);
    
    std::mt19937 rng(42); // Fixed seed for reproducibility
    std::uniform_int_distribution<uint32_t> xDist(0, field.nx() - 1);
    std::uniform_int_distribution<uint32_t> yDist(0, field.ny() - 1);
    
    for (size_t i = 0; i < numAccesses; ++i) {
        randomIndices.emplace_back(xDist(rng), yDist(rng));
    }
    
    // Measure sequential access time (row-major order)
    Timer seqTimer;
    double seqSum = 0.0;
    
    // Repeat accesses to get measurable time
    for (size_t rep = 0; rep < 5; ++rep) {
        for (uint32_t j = 0; j < field.ny(); ++j) {
            for (uint32_t i = 0; i < field.nx(); ++i) {
                seqSum += field(i, j);
            }
        }
    }
    
    double seqTime = seqTimer.elapsed();
    
    // Measure random access time
    Timer randTimer;
    double randSum = 0.0;
    
    for (size_t i = 0; i < numAccesses; ++i) {
        const auto& [x, y] = randomIndices[i];
        randSum += field(x, y);
    }
    
    double randTime = randTimer.elapsed();
    
    // Measure direct memory access time (baseline)
    Timer directTimer;
    double directSum = 0.0;
    double* data = field.data();
    
    for (size_t i = 0; i < field.size(); ++i) {
        directSum += data[i];
    }
    
    double directTime = directTimer.elapsed();
    
    // Print performance results (for informational purposes)
    std::cout << "Sequential access time: " << seqTime << " ms" << std::endl;
    std::cout << "Random access time: " << randTime << " ms" << std::endl;
    std::cout << "Direct memory access time: " << directTime << " ms" << std::endl;
    
    // We expect random access to be slower than sequential access
    // But this is not a strict requirement, so we don't assert it
    
    // Ensure computations weren't optimized away by using the sums
    EXPECT_EQ(seqSum, 0.0);
    EXPECT_EQ(randSum, 0.0);
    EXPECT_EQ(directSum, 0.0);
}

// Test 2: Performance with different storage layouts
TEST_F(FieldPerformanceTest, StorageLayoutPerformance) {
    // Create meshes with different storage layouts
    class BlockedMesh : public mesh::Mesh {
    public:
        BlockedMesh(uint32_t nx, uint32_t ny, int blockSize) 
            : mesh::Mesh(nx, ny), m_blockSize(blockSize) {}
        
        uint32_t linearIndex(uint32_t i, uint32_t j) const override {
            return field::storage::blocked(*this, i, j, m_blockSize);
        }
        
    private:
        int m_blockSize;
    };
    
    class ZOrderMesh : public mesh::Mesh {
    public:
        ZOrderMesh(uint32_t nx, uint32_t ny) : mesh::Mesh(nx, ny) {}
        
        uint32_t linearIndex(uint32_t i, uint32_t j) const override {
            return field::storage::zOrder(*this, i, j);
        }
    };
    
    // Create test meshes (smaller size for quicker tests)
    mesh::Mesh rowMajorMesh(64, 64);
    BlockedMesh blockedMesh(64, 64, 8);
    ZOrderMesh zOrderMesh(64, 64);
    
    // Create fields with the different meshes
    mesh::Field<double> rowMajorField(rowMajorMesh, 1.0);
    mesh::Field<double> blockedField(blockedMesh, 1.0);
    mesh::Field<double> zOrderField(zOrderMesh, 1.0);
    
    // Fill the fields with test data
    for (uint32_t j = 0; j < rowMajorField.ny(); ++j) {
        for (uint32_t i = 0; i < rowMajorField.nx(); ++i) {
            double value = i + j * 0.1;
            rowMajorField(i, j) = value;
            blockedField(i, j) = value;
            zOrderField(i, j) = value;
        }
    }
    
    // Test row-major traversal performance
    Timer rowMajorTimer;
    double rowMajorSum = 0.0;
    
    for (uint32_t j = 0; j < rowMajorField.ny(); ++j) {
        for (uint32_t i = 0; i < rowMajorField.nx(); ++i) {
            rowMajorSum += rowMajorField(i, j);
        }
    }
    
    double rowMajorTime = rowMajorTimer.elapsed();
    
    // Test blocked traversal performance
    Timer blockedTimer;
    double blockedSum = 0.0;
    
    for (uint32_t j = 0; j < blockedField.ny(); ++j) {
        for (uint32_t i = 0; i < blockedField.nx(); ++i) {
            blockedSum += blockedField(i, j);
        }
    }
    
    double blockedTime = blockedTimer.elapsed();
    
    // Test Z-order traversal performance
    Timer zOrderTimer;
    double zOrderSum = 0.0;
    
    for (uint32_t j = 0; j < zOrderField.ny(); ++j) {
        for (uint32_t i = 0; i < zOrderField.nx(); ++i) {
            zOrderSum += zOrderField(i, j);
        }
    }
    
    double zOrderTime = zOrderTimer.elapsed();
    
    // Print performance results
    std::cout << "Row-major traversal time: " << rowMajorTime << " ms" << std::endl;
    std::cout << "Blocked traversal time: " << blockedTime << " ms" << std::endl;
    std::cout << "Z-order traversal time: " << zOrderTime << " ms" << std::endl;
    
    // Ensure the sums are approximately equal (floating point precision issues)
    EXPECT_NEAR(rowMajorSum, blockedSum, 1e-10);
    EXPECT_NEAR(rowMajorSum, zOrderSum, 1e-10);
}

// Test 3: Performance comparison of different data types
TEST_F(FieldPerformanceTest, DataTypePerformance) {
    // Create fields with different data types
    mesh::Field<int> intField(*mesh, 0);
    mesh::Field<double> doubleField(*mesh, 0.0);
    mesh::Field<std::pair<float, float>> vectorField(*mesh, {0.0f, 0.0f});
    
    const int numIterations = 10;
    
    // Test int field performance
    Timer intTimer;
    int intSum = 0;
    
    for (int iter = 0; iter < numIterations; ++iter) {
        for (uint32_t j = 0; j < intField.ny(); ++j) {
            for (uint32_t i = 0; i < intField.nx(); ++i) {
                intField(i, j) = i + j;
                intSum += intField(i, j);
            }
        }
    }
    
    double intTime = intTimer.elapsed();
    
    // Test double field performance
    Timer doubleTimer;
    double doubleSum = 0.0;
    
    for (int iter = 0; iter < numIterations; ++iter) {
        for (uint32_t j = 0; j < doubleField.ny(); ++j) {
            for (uint32_t i = 0; i < doubleField.nx(); ++i) {
                doubleField(i, j) = i + j * 0.1;
                doubleSum += doubleField(i, j);
            }
        }
    }
    
    double doubleTime = doubleTimer.elapsed();
    
    // Test vector field performance
    Timer vectorTimer;
    float vectorSum = 0.0f;
    
    for (int iter = 0; iter < numIterations; ++iter) {
        for (uint32_t j = 0; j < vectorField.ny(); ++j) {
            for (uint32_t i = 0; i < vectorField.nx(); ++i) {
                vectorField(i, j) = {static_cast<float>(i), static_cast<float>(j)};
                vectorSum += vectorField(i, j).first + vectorField(i, j).second;
            }
        }
    }
    
    double vectorTime = vectorTimer.elapsed();
    
    // Print performance results
    std::cout << "Int field time: " << intTime << " ms" << std::endl;
    std::cout << "Double field time: " << doubleTime << " ms" << std::endl;
    std::cout << "Vector field time: " << vectorTime << " ms" << std::endl;
    
    // Ensure the computations weren't optimized away
    EXPECT_NE(intSum, 0);
    EXPECT_NE(doubleSum, 0.0);
    EXPECT_NE(vectorSum, 0.0f);
}

// Test 4: Performance of always_inline attribute
TEST_F(FieldPerformanceTest, InliningPerformance) {
    mesh::Field<double> field(*mesh, 0.0);
    
    // Explicitly inlined function
    auto inlined = [](mesh::Field<double>& f, uint32_t i, uint32_t j) __attribute__((always_inline)) {
        return f(i, j);
    };
    
    // Regular function (may or may not be inlined by the compiler)
    auto regular = [](mesh::Field<double>& f, uint32_t i, uint32_t j) {
        return f(i, j);
    };
    
    const int numIterations = 100;
    
    // Test explicitly inlined access
    Timer inlinedTimer;
    double inlinedSum = 0.0;
    
    for (int iter = 0; iter < numIterations; ++iter) {
        for (uint32_t j = 0; j < field.ny(); ++j) {
            for (uint32_t i = 0; i < field.nx(); ++i) {
                inlinedSum += inlined(field, i, j);
            }
        }
    }
    
    double inlinedTime = inlinedTimer.elapsed();
    
    // Test regular function access
    Timer regularTimer;
    double regularSum = 0.0;
    
    for (int iter = 0; iter < numIterations; ++iter) {
        for (uint32_t j = 0; j < field.ny(); ++j) {
            for (uint32_t i = 0; i < field.nx(); ++i) {
                regularSum += regular(field, i, j);
            }
        }
    }
    
    double regularTime = regularTimer.elapsed();
    
    // Print performance results
    std::cout << "Explicitly inlined access time: " << inlinedTime << " ms" << std::endl;
    std::cout << "Regular function access time: " << regularTime << " ms" << std::endl;
    
    // Ensure the sums match
    EXPECT_DOUBLE_EQ(inlinedSum, regularSum);
    
    // Note: We don't assert on relative performance as compiler optimizations
    // may eliminate any differences between the two approaches
}

// Test 5: Cache-aware iteration patterns
TEST_F(FieldPerformanceTest, CacheAwareIteration) {
    mesh::Field<double> field(*mesh, 1.0);
    const int repeats = 10;
    
    // Row-major iteration (i varies fastest)
    Timer rowMajorTimer;
    double rowMajorSum = 0.0;
    
    for (int rep = 0; rep < repeats; ++rep) {
        for (uint32_t j = 0; j < field.ny(); ++j) {
            for (uint32_t i = 0; i < field.nx(); ++i) {
                rowMajorSum += field(i, j);
            }
        }
    }
    
    double rowMajorTime = rowMajorTimer.elapsed();
    
    // Column-major iteration (j varies fastest) - less cache friendly
    Timer colMajorTimer;
    double colMajorSum = 0.0;
    
    for (int rep = 0; rep < repeats; ++rep) {
        for (uint32_t i = 0; i < field.nx(); ++i) {
            for (uint32_t j = 0; j < field.ny(); ++j) {
                colMajorSum += field(i, j);
            }
        }
    }
    
    double colMajorTime = colMajorTimer.elapsed();
    
    // Blocked iteration (improve cache locality)
    Timer blockedTimer;
    double blockedSum = 0.0;
    constexpr int blockSize = 32; // Tuned for common cache line sizes
    
    for (int rep = 0; rep < repeats; ++rep) {
        for (uint32_t jBlock = 0; jBlock < field.ny(); jBlock += blockSize) {
            for (uint32_t iBlock = 0; iBlock < field.nx(); iBlock += blockSize) {
                for (uint32_t j = jBlock; j < std::min(jBlock + blockSize, field.ny()); ++j) {
                    for (uint32_t i = iBlock; i < std::min(iBlock + blockSize, field.nx()); ++i) {
                        blockedSum += field(i, j);
                    }
                }
            }
        }
    }
    
    double blockedTime = blockedTimer.elapsed();
    
    // Print performance results
    std::cout << "Row-major iteration time: " << rowMajorTime << " ms" << std::endl;
    std::cout << "Column-major iteration time: " << colMajorTime << " ms" << std::endl;
    std::cout << "Blocked iteration time: " << blockedTime << " ms" << std::endl;
    
    // Verify all sums are equal
    EXPECT_DOUBLE_EQ(rowMajorSum, colMajorSum);
    EXPECT_DOUBLE_EQ(rowMajorSum, blockedSum);
    
    // Row-major should generally be faster than column-major due to cache behavior
    // but we don't assert this as compiler optimizations may eliminate the difference
}

// Test 6: Testing the performance of Field with stack vs heap allocation
TEST_F(FieldPerformanceTest, StackVsHeapAllocation) {
    // Setup - create a smaller mesh for this test
    mesh::Mesh smallMesh(16, 16); // Small enough to fit on stack
    const int iterations = 10000;
    
    // Stack allocation timing
    Timer stackTimer;
    double stackSum = 0.0;
    
    for (int i = 0; i < iterations; ++i) {
        // Field is allocated on the stack in each iteration
        mesh::Field<double> stackField(smallMesh, 1.0);
        stackSum += stackField(0, 0);
    }
    
    double stackTime = stackTimer.elapsed();
    
    // Heap allocation timing
    Timer heapTimer;
    double heapSum = 0.0;
    
    for (int i = 0; i < iterations; ++i) {
        // Field is allocated on the heap in each iteration
        auto* heapField = new mesh::Field<double>(smallMesh, 1.0);
        heapSum += (*heapField)(0, 0);
        delete heapField;
    }
    
    double heapTime = heapTimer.elapsed();
    
    // Print performance results
    std::cout << "Stack allocation time: " << stackTime << " ms" << std::endl;
    std::cout << "Heap allocation time: " << heapTime << " ms" << std::endl;
    
    // Verify both methods give same results
    EXPECT_DOUBLE_EQ(stackSum, heapSum);
}

// Test 7: Test the overhead of the Field interface vs raw array
TEST_F(FieldPerformanceTest, FieldVsRawArrayOverhead) {
    // Create a field and an equivalent raw array
    mesh::Field<double> field(*mesh, 0.0);
    std::vector<double> rawArray(field.size(), 0.0);
    
    const int iterations = 10;
    
    // Time field access using operator()
    Timer fieldTimer;
    double fieldSum = 0.0;
    
    for (int iter = 0; iter < iterations; ++iter) {
        for (uint32_t j = 0; j < field.ny(); ++j) {
            for (uint32_t i = 0; i < field.nx(); ++i) {
                field(i, j) = i + j;
                fieldSum += field(i, j);
            }
        }
    }
    
    double fieldTime = fieldTimer.elapsed();
    
    // Time raw array access with manual index calculation
    Timer rawTimer;
    double rawSum = 0.0;
    
    for (int iter = 0; iter < iterations; ++iter) {
        for (uint32_t j = 0; j < field.ny(); ++j) {
            for (uint32_t i = 0; i < field.nx(); ++i) {
                size_t idx = i + j * field.nx();
                rawArray[idx] = i + j;
                rawSum += rawArray[idx];
            }
        }
    }
    
    double rawTime = rawTimer.elapsed();
    
    // Time field access using direct array access
    Timer directTimer;
    double directSum = 0.0;
    
    for (int iter = 0; iter < iterations; ++iter) {
        double* data = field.data();
        for (uint32_t j = 0; j < field.ny(); ++j) {
            for (uint32_t i = 0; i < field.nx(); ++i) {
                size_t idx = i + j * field.nx();
                data[idx] = i + j;
                directSum += data[idx];
            }
        }
    }
    
    double directTime = directTimer.elapsed();
    
    // Print performance results
    std::cout << "Field operator() access time: " << fieldTime << " ms" << std::endl;
    std::cout << "Raw array access time: " << rawTime << " ms" << std::endl;
    std::cout << "Field direct data access time: " << directTime << " ms" << std::endl;
    
    // Verify all sums are equal
    EXPECT_DOUBLE_EQ(fieldSum, rawSum);
    EXPECT_DOUBLE_EQ(fieldSum, directSum);
}

// Test 8: Testing the performance impact of different field sizes
TEST_F(FieldPerformanceTest, FieldSizeImpact) {
    // Create meshes and fields of different sizes
    mesh::Mesh smallMesh(32, 32);
    mesh::Mesh mediumMesh(128, 128);
    mesh::Mesh largeMesh(512, 512);
    
    mesh::Field<double> smallField(smallMesh, 1.0);
    mesh::Field<double> mediumField(mediumMesh, 1.0);
    mesh::Field<double> largeField(largeMesh, 1.0);
    
    const int smallIterations = 100;
    const int mediumIterations = 10;
    const int largeIterations = 1;
    
    // Time for small field
    Timer smallTimer;
    double smallSum = 0.0;
    
    for (int iter = 0; iter < smallIterations; ++iter) {
        for (uint32_t j = 0; j < smallField.ny(); ++j) {
            for (uint32_t i = 0; i < smallField.nx(); ++i) {
                smallField(i, j) = i + j;
                smallSum += smallField(i, j);
            }
        }
    }
    
    double smallTime = smallTimer.elapsed() / smallIterations;
    
    // Time for medium field
    Timer mediumTimer;
    double mediumSum = 0.0;
    
    for (int iter = 0; iter < mediumIterations; ++iter) {
        for (uint32_t j = 0; j < mediumField.ny(); ++j) {
            for (uint32_t i = 0; i < mediumField.nx(); ++i) {
                mediumField(i, j) = i + j;
                mediumSum += mediumField(i, j);
            }
        }
    }
    
    double mediumTime = mediumTimer.elapsed() / mediumIterations;
    
    // Time for large field
    Timer largeTimer;
    double largeSum = 0.0;
    
    for (int iter = 0; iter < largeIterations; ++iter) {
        for (uint32_t j = 0; j < largeField.ny(); ++j) {
            for (uint32_t i = 0; i < largeField.nx(); ++i) {
                largeField(i, j) = i + j;
                largeSum += largeField(i, j);
            }
        }
    }
    
    double largeTime = largeTimer.elapsed() / largeIterations;
    
    // Print performance results
    std::cout << "Small field (32x32) time per iteration: " << smallTime << " ms" << std::endl;
    std::cout << "Medium field (128x128) time per iteration: " << mediumTime << " ms" << std::endl;
    std::cout << "Large field (512x512) time per iteration: " << largeTime << " ms" << std::endl;
    
    // Ensure the computations weren't optimized away
    EXPECT_NE(smallSum, 0.0);
    EXPECT_NE(mediumSum, 0.0);
    EXPECT_NE(largeSum, 0.0);
    
    // Verify scaling behavior (should be roughly O(n²))
    // But this is a loose constraint due to caching effects
    double smallToMediumRatio = mediumTime / smallTime;
    double mediumToLargeRatio = largeTime / mediumTime;
    
    // Expected ratios for perfect O(n²) scaling would be 16 (4²) for each step
    // We use a loose bound to account for caching and other effects
    EXPECT_GT(smallToMediumRatio, 4.0);  // Should at least scale with n
    EXPECT_GT(mediumToLargeRatio, 4.0);  // Should at least scale with n
}