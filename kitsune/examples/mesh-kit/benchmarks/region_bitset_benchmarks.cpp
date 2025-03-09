/**
 * @file RegionBenchmark.cpp
 * @brief Performance benchmarks for Region class optimizations
 * 
 * This benchmark suite compares the performance of the optimized BitArray 
 * implementation against std::vector<bool> for common region operations.
 * 
 * Build with:
 * g++ -std=c++17 -O3 -DNDEBUG RegionBenchmark.cpp -lbenchmark -lbenchmark_main -lpthread -o region_benchmark
 */

#include <benchmark/benchmark.h>
#include <vector>
#include <random>
#include <unordered_set>
#include <algorithm>
#include <execution>
#include "BitArray.h"

// Size constants for benchmarks
constexpr size_t SMALL_MESH_SIZE = 1'000;        // 1K cells (32x32 mesh)
constexpr size_t MEDIUM_MESH_SIZE = 100'000;     // 100K cells (316x316 mesh)
constexpr size_t LARGE_MESH_SIZE = 1'000'000;    // 1M cells (1000x1000 mesh)
constexpr size_t VERY_LARGE_MESH_SIZE = 10'000'000; // 10M cells (3162x3162 mesh)

// Density constants for different region types
constexpr double SPARSE_DENSITY = 0.01;           // 1% of mesh (e.g., small feature)
constexpr double MEDIUM_DENSITY = 0.1;            // 10% of mesh (e.g., boundary layer)
constexpr double HIGH_DENSITY = 0.5;              // 50% of mesh (e.g., material region)

/**
 * @brief Generate random indices for a region
 * 
 * @param count Number of indices to generate
 * @param maxIndex Maximum index value (exclusive)
 * @return Vector of unique random indices
 */
std::vector<size_t> generateRandomIndices(size_t count, size_t maxIndex) {
    std::unordered_set<size_t> uniqueIndices;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dist(0, maxIndex - 1);
    
    // Generate unique indices
    while (uniqueIndices.size() < count && uniqueIndices.size() < maxIndex) {
        uniqueIndices.insert(dist(gen));
    }
    
    return std::vector<size_t>(uniqueIndices.begin(), uniqueIndices.end());
}

/**
 * @brief Create a random bit pattern
 * 
 * @param size Size of the bit container
 * @param density Fraction of bits that should be set
 * @return Vector of indices where bits are set
 */
std::vector<size_t> createRandomBitPattern(size_t size, double density) {
    size_t setBitsCount = static_cast<size_t>(size * density);
    return generateRandomIndices(setBitsCount, size);
}

/**
 * @brief Initialize a std::vector<bool> with the given pattern
 * 
 * @param size Size of the vector
 * @param setIndices Indices where bits should be set
 * @return Initialized vector
 */
std::vector<bool> initializeStdVector(size_t size, const std::vector<size_t>& setIndices) {
    std::vector<bool> vec(size, false);
    for (size_t idx : setIndices) {
        vec[idx] = true;
    }
    return vec;
}

/**
 * @brief Initialize a BitArray with the given pattern
 * 
 * @param size Size of the bit array
 * @param setIndices Indices where bits should be set
 * @return Initialized bit array
 */
BitArray initializeBitArray(size_t size, const std::vector<size_t>& setIndices) {
    BitArray bits(size, false);
    for (size_t idx : setIndices) {
        bits.set(idx, true);
    }
    return bits;
}

//----------------------------------------------------------
// Bit Setting/Getting Benchmarks
//----------------------------------------------------------

/**
 * @brief Benchmark setting individual bits in std::vector<bool>
 */
static void BM_StdVector_SetBits(benchmark::State& state) {
    size_t size = state.range(0);
    double density = static_cast<double>(state.range(1)) / 100.0;
    
    for (auto _ : state) {
        state.PauseTiming();
        auto indices = createRandomBitPattern(size, density);
        std::vector<bool> vec(size, false);
        state.ResumeTiming();
        
        for (size_t idx : indices) {
            vec[idx] = true;
        }
    }
    
    state.SetComplexityN(state.range(0));
    state.SetItemsProcessed(state.iterations() * size * density);
}

/**
 * @brief Benchmark setting individual bits in BitArray
 */
static void BM_BitArray_SetBits(benchmark::State& state) {
    size_t size = state.range(0);
    double density = static_cast<double>(state.range(1)) / 100.0;
    
    for (auto _ : state) {
        state.PauseTiming();
        auto indices = createRandomBitPattern(size, density);
        BitArray bits(size, false);
        state.ResumeTiming();
        
        for (size_t idx : indices) {
            bits.set(idx, true);
        }
    }
    
    state.SetComplexityN(state.range(0));
    state.SetItemsProcessed(state.iterations() * size * density);
}

/**
 * @brief Benchmark reading individual bits from std::vector<bool>
 */
static void BM_StdVector_GetBits(benchmark::State& state) {
    size_t size = state.range(0);
    double density = static_cast<double>(state.range(1)) / 100.0;
    
    // Setup
    auto indices = createRandomBitPattern(size, density);
    std::vector<bool> vec = initializeStdVector(size, indices);
    size_t checkIndices = std::min(size, static_cast<size_t>(1000)); // Check at most 1000 indices
    std::vector<size_t> queryIndices = generateRandomIndices(checkIndices, size);
    
    // Prevent optimization
    volatile bool result = false;
    
    for (auto _ : state) {
        for (size_t idx : queryIndices) {
            result = vec[idx];
            benchmark::DoNotOptimize(result);
        }
    }
    
    state.SetComplexityN(state.range(0));
    state.SetItemsProcessed(state.iterations() * queryIndices.size());
}

/**
 * @brief Benchmark reading individual bits from BitArray
 */
static void BM_BitArray_GetBits(benchmark::State& state) {
    size_t size = state.range(0);
    double density = static_cast<double>(state.range(1)) / 100.0;
    
    // Setup
    auto indices = createRandomBitPattern(size, density);
    BitArray bits = initializeBitArray(size, indices);
    size_t checkIndices = std::min(size, static_cast<size_t>(1000)); // Check at most 1000 indices
    std::vector<size_t> queryIndices = generateRandomIndices(checkIndices, size);
    
    // Prevent optimization
    volatile bool result = false;
    
    for (auto _ : state) {
        for (size_t idx : queryIndices) {
            result = bits.get(idx);
            benchmark::DoNotOptimize(result);
        }
    }
    
    state.SetComplexityN(state.range(0));
    state.SetItemsProcessed(state.iterations() * queryIndices.size());
}

//----------------------------------------------------------
// Count Benchmarks (Population Count)
//----------------------------------------------------------

/**
 * @brief Benchmark counting set bits in std::vector<bool>
 */
static void BM_StdVector_CountBits(benchmark::State& state) {
    size_t size = state.range(0);
    double density = static_cast<double>(state.range(1)) / 100.0;
    
    // Setup
    auto indices = createRandomBitPattern(size, density);
    std::vector<bool> vec = initializeStdVector(size, indices);
    
    for (auto _ : state) {
        size_t count = 0;
        for (bool bit : vec) {
            if (bit) count++;
        }
        benchmark::DoNotOptimize(count);
    }
    
    state.SetComplexityN(state.range(0));
    state.SetItemsProcessed(state.iterations() * size);
}

/**
 * @brief Benchmark counting set bits in BitArray
 */
static void BM_BitArray_CountBits(benchmark::State& state) {
    size_t size = state.range(0);
    double density = static_cast<double>(state.range(1)) / 100.0;
    
    // Setup
    auto indices = createRandomBitPattern(size, density);
    BitArray bits = initializeBitArray(size, indices);
    
    for (auto _ : state) {
        size_t count = bits.count();
        benchmark::DoNotOptimize(count);
    }
    
    state.SetComplexityN(state.range(0));
    state.SetItemsProcessed(state.iterations() * size);
}

//----------------------------------------------------------
// Set Operation Benchmarks
//----------------------------------------------------------

/**
 * @brief Benchmark union operation with std::vector<bool>
 */
static void BM_StdVector_Union(benchmark::State& state) {
    size_t size = state.range(0);
    double densityA = static_cast<double>(state.range(1)) / 100.0;
    double densityB = static_cast<double>(state.range(1)) / 100.0;
    
    // Setup
    auto indicesA = createRandomBitPattern(size, densityA);
    auto indicesB = createRandomBitPattern(size, densityB);
    std::vector<bool> vecA = initializeStdVector(size, indicesA);
    std::vector<bool> vecB = initializeStdVector(size, indicesB);
    
    for (auto _ : state) {
        std::vector<bool> result(size, false);
        for (size_t i = 0; i < size; ++i) {
            result[i] = vecA[i] || vecB[i];
        }
        benchmark::DoNotOptimize(result);
    }
    
    state.SetComplexityN(state.range(0));
    state.SetItemsProcessed(state.iterations() * size);
}

/**
 * @brief Benchmark union operation with BitArray
 */
static void BM_BitArray_Union(benchmark::State& state) {
    size_t size = state.range(0);
    double densityA = static_cast<double>(state.range(1)) / 100.0;
    double densityB = static_cast<double>(state.range(1)) / 100.0;
    
    // Setup
    auto indicesA = createRandomBitPattern(size, densityA);
    auto indicesB = createRandomBitPattern(size, densityB);
    BitArray bitsA = initializeBitArray(size, indicesA);
    BitArray bitsB = initializeBitArray(size, indicesB);
    
    for (auto _ : state) {
        BitArray result = bitsA;  // Make a copy
        result.bitwiseOr(bitsB);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetComplexityN(state.range(0));
    state.SetItemsProcessed(state.iterations() * size);
}

/**
 * @brief Benchmark intersection operation with std::vector<bool>
 */
static void BM_StdVector_Intersection(benchmark::State& state) {
    size_t size = state.range(0);
    double densityA = static_cast<double>(state.range(1)) / 100.0;
    double densityB = static_cast<double>(state.range(1)) / 100.0;
    
    // Setup
    auto indicesA = createRandomBitPattern(size, densityA);
    auto indicesB = createRandomBitPattern(size, densityB);
    std::vector<bool> vecA = initializeStdVector(size, indicesA);
    std::vector<bool> vecB = initializeStdVector(size, indicesB);
    
    for (auto _ : state) {
        std::vector<bool> result(size, false);
        for (size_t i = 0; i < size; ++i) {
            result[i] = vecA[i] && vecB[i];
        }
        benchmark::DoNotOptimize(result);
    }
    
    state.SetComplexityN(state.range(0));
    state.SetItemsProcessed(state.iterations() * size);
}

/**
 * @brief Benchmark intersection operation with BitArray
 */
static void BM_BitArray_Intersection(benchmark::State& state) {
    size_t size = state.range(0);
    double densityA = static_cast<double>(state.range(1)) / 100.0;
    double densityB = static_cast<double>(state.range(1)) / 100.0;
    
    // Setup
    auto indicesA = createRandomBitPattern(size, densityA);
    auto indicesB = createRandomBitPattern(size, densityB);
    BitArray bitsA = initializeBitArray(size, indicesA);
    BitArray bitsB = initializeBitArray(size, indicesB);
    
    for (auto _ : state) {
        BitArray result = bitsA;  // Make a copy
        result.bitwiseAnd(bitsB);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetComplexityN(state.range(0));
    state.SetItemsProcessed(state.iterations() * size);
}

/**
 * @brief Benchmark difference operation with std::vector<bool>
 */
static void BM_StdVector_Difference(benchmark::State& state) {
    size_t size = state.range(0);
    double densityA = static_cast<double>(state.range(1)) / 100.0;
    double densityB = densityA / 2.0;  // Make B subset of A for realistic difference
    
    // Setup
    auto indicesA = createRandomBitPattern(size, densityA);
    auto indicesB = createRandomBitPattern(size, densityB);
    std::vector<bool> vecA = initializeStdVector(size, indicesA);
    std::vector<bool> vecB = initializeStdVector(size, indicesB);
    
    for (auto _ : state) {
        std::vector<bool> result(size, false);
        for (size_t i = 0; i < size; ++i) {
            result[i] = vecA[i] && !vecB[i];
        }
        benchmark::DoNotOptimize(result);
    }
    
    state.SetComplexityN(state.range(0));
    state.SetItemsProcessed(state.iterations() * size);
}

/**
 * @brief Benchmark difference operation with BitArray
 */
static void BM_BitArray_Difference(benchmark::State& state) {
    size_t size = state.range(0);
    double densityA = static_cast<double>(state.range(1)) / 100.0;
    double densityB = densityA / 2.0;  // Make B subset of A for realistic difference
    
    // Setup
    auto indicesA = createRandomBitPattern(size, densityA);
    auto indicesB = createRandomBitPattern(size, densityB);
    BitArray bitsA = initializeBitArray(size, indicesA);
    BitArray bitsB = initializeBitArray(size, indicesB);
    
    for (auto _ : state) {
        BitArray result = bitsA;  // Make a copy
        result.bitwiseAndNot(bitsB);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetComplexityN(state.range(0));
    state.SetItemsProcessed(state.iterations() * size);
}

//----------------------------------------------------------
// Iteration Benchmarks
//----------------------------------------------------------

/**
 * @brief Benchmark finding all set bits in std::vector<bool>
 */
static void BM_StdVector_FindBits(benchmark::State& state) {
    size_t size = state.range(0);
    double density = static_cast<double>(state.range(1)) / 100.0;
    
    // Setup
    auto indices = createRandomBitPattern(size, density);
    std::vector<bool> vec = initializeStdVector(size, indices);
    
    for (auto _ : state) {
        std::vector<size_t> result;
        result.reserve(indices.size());
        
        for (size_t i = 0; i < size; ++i) {
            if (vec[i]) {
                result.push_back(i);
            }
        }
        benchmark::DoNotOptimize(result);
    }
    
    state.SetComplexityN(state.range(0));
    state.SetItemsProcessed(state.iterations() * size);
}

/**
 * @brief Benchmark finding all set bits in BitArray
 */
static void BM_BitArray_FindBits(benchmark::State& state) {
    size_t size = state.range(0);
    double density = static_cast<double>(state.range(1)) / 100.0;
    
    // Setup
    auto indices = createRandomBitPattern(size, density);
    BitArray bits = initializeBitArray(size, indices);
    
    for (auto _ : state) {
        std::vector<size_t> result;
        result.reserve(indices.size());
        
        size_t idx = bits.findFirst();
        while (idx < bits.size()) {
            result.push_back(idx);
            idx = bits.findNext(idx);
        }
        benchmark::DoNotOptimize(result);
    }
    
    state.SetComplexityN(state.range(0));
    state.SetItemsProcessed(state.iterations() * size);
}

//----------------------------------------------------------
// Real-World Computational Workloads
//----------------------------------------------------------

/**
 * @brief Benchmark finding a boundary region with std::vector<bool>
 * 
 * This simulates finding the boundary of a region by checking
 * if any neighboring cell is not in the region.
 */
static void BM_StdVector_BoundaryExtraction(benchmark::State& state) {
    size_t size = state.range(0);
    double density = static_cast<double>(state.range(1)) / 100.0;
    
    // Setup: create a grid with a central region
    size_t width = static_cast<size_t>(std::sqrt(size));
    size_t height = width;
    
    // Create a circular region at the center
    std::vector<bool> region(width * height, false);
    size_t centerX = width / 2;
    size_t centerY = height / 2;
    double radius = std::sqrt(size * density / M_PI);
    
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            double dx = static_cast<double>(x) - centerX;
            double dy = static_cast<double>(y) - centerY;
            double distance = std::sqrt(dx*dx + dy*dy);
            
            if (distance <= radius) {
                region[y * width + x] = true;
            }
        }
    }
    
    for (auto _ : state) {
        // Find the boundary of the region
        std::vector<bool> boundary(width * height, false);
        
        for (size_t y = 1; y < height - 1; ++y) {
            for (size_t x = 1; x < width - 1; ++x) {
                size_t idx = y * width + x;
                
                if (region[idx]) {
                    // Check if any neighbor is outside the region
                    bool isBoundary = 
                        !region[(y-1) * width + x] ||    // North
                        !region[y * width + (x+1)] ||    // East
                        !region[(y+1) * width + x] ||    // South
                        !region[y * width + (x-1)];      // West
                    
                    boundary[idx] = isBoundary;
                }
            }
        }
        
        benchmark::DoNotOptimize(boundary);
    }
    
    state.SetComplexityN(state.range(0));
    state.SetItemsProcessed(state.iterations() * size);
}

/**
 * @brief Benchmark finding a boundary region with BitArray
 */
static void BM_BitArray_BoundaryExtraction(benchmark::State& state) {
    size_t size = state.range(0);
    double density = static_cast<double>(state.range(1)) / 100.0;
    
    // Setup: create a grid with a central region
    size_t width = static_cast<size_t>(std::sqrt(size));
    size_t height = width;
    
    // Create a circular region at the center
    BitArray region(width * height, false);
    size_t centerX = width / 2;
    size_t centerY = height / 2;
    double radius = std::sqrt(size * density / M_PI);
    
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            double dx = static_cast<double>(x) - centerX;
            double dy = static_cast<double>(y) - centerY;
            double distance = std::sqrt(dx*dx + dy*dy);
            
            if (distance <= radius) {
                region.set(y * width + x, true);
            }
        }
    }
    
    for (auto _ : state) {
        // Find the boundary of the region
        BitArray boundary(width * height, false);
        
        for (size_t y = 1; y < height - 1; ++y) {
            for (size_t x = 1; x < width - 1; ++x) {
                size_t idx = y * width + x;
                
                if (region.get(idx)) {
                    // Check if any neighbor is outside the region
                    bool isBoundary = 
                        !region.get((y-1) * width + x) ||    // North
                        !region.get(y * width + (x+1)) ||    // East
                        !region.get((y+1) * width + x) ||    // South
                        !region.get(y * width + (x-1));      // West
                    
                    boundary.set(idx, isBoundary);
                }
            }
        }
        
        benchmark::DoNotOptimize(boundary);
    }
    
    state.SetComplexityN(state.range(0));
    state.SetItemsProcessed(state.iterations() * size);
}

/**
 * @brief Benchmark adaptive mesh refinement with std::vector<bool>
 * 
 * This simulates refinement of a region based on a gradient criteria.
 */
static void BM_StdVector_AdaptiveRefinement(benchmark::State& state) {
    size_t size = state.range(0);
    double density = static_cast<double>(state.range(1)) / 100.0;
    
    // Setup: create a grid with a gradient field
    size_t width = static_cast<size_t>(std::sqrt(size));
    size_t height = width;
    
    // Create a scalar field with a steep gradient in the center
    std::vector<double> field(width * height);
    size_t centerX = width / 2;
    size_t centerY = height / 2;
    
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            double dx = static_cast<double>(x) - centerX;
            double dy = static_cast<double>(y) - centerY;
            double distance = std::sqrt(dx*dx + dy*dy);
            
            // Gaussian function
            field[y * width + x] = std::exp(-distance*distance / (2.0 * width * density));
        }
    }
    
    for (auto _ : state) {
        // Mark cells for refinement where gradient is high
        std::vector<bool> refine(width * height, false);
        
        for (size_t y = 1; y < height - 1; ++y) {
            for (size_t x = 1; x < width - 1; ++x) {
                size_t idx = y * width + x;
                
                // Compute gradient magnitude (central difference)
                double gradX = (field[y * width + (x+1)] - field[y * width + (x-1)]) / 2.0;
                double gradY = (field[(y+1) * width + x] - field[(y-1) * width + x]) / 2.0;
                double gradMag = std::sqrt(gradX*gradX + gradY*gradY);
                
                // Mark for refinement if gradient is above threshold
                refine[idx] = (gradMag > 0.01);
            }
        }
        
        // Expand refinement to include neighboring cells
        std::vector<bool> expandedRefine = refine;
        
        for (size_t y = 1; y < height - 1; ++y) {
            for (size_t x = 1; x < width - 1; ++x) {
                size_t idx = y * width + x;
                
                if (refine[idx]) {
                    // Mark neighboring cells
                    expandedRefine[(y-1) * width + x] = true;    // North
                    expandedRefine[y * width + (x+1)] = true;    // East
                    expandedRefine[(y+1) * width + x] = true;    // South
                    expandedRefine[y * width + (x-1)] = true;    // West
                }
            }
        }
        
        benchmark::DoNotOptimize(expandedRefine);
    }
    
    state.SetComplexityN(state.range(0));
    state.SetItemsProcessed(state.iterations() * size);
}

/**
 * @brief Benchmark adaptive mesh refinement with BitArray
 */
static void BM_BitArray_AdaptiveRefinement(benchmark::State& state) {
    size_t size = state.range(0);
    double density = static_cast<double>(state.range(1)) / 100.0;
    
    // Setup: create a grid with a gradient field
    size_t width = static_cast<size_t>(std::sqrt(size));
    size_t height = width;
    
    // Create a scalar field with a steep gradient in the center
    std::vector<double> field(width * height);
    size_t centerX = width / 2;
    size_t centerY = height / 2;
    
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            double dx = static_cast<double>(x) - centerX;
            double dy = static_cast<double>(y) - centerY;
            double distance = std::sqrt(dx*dx + dy*dy);
            
            // Gaussian function
            field[y * width + x] = std::exp(-distance*distance / (2.0 * width * density));
        }
    }
    
    for (auto _ : state) {
        // Mark cells for refinement where gradient is high
        BitArray refine(width * height, false);
        
        for (size_t y = 1; y < height - 1; ++y) {
            for (size_t x = 1; x < width - 1; ++x) {
                size_t idx = y * width + x;
                
                // Compute gradient magnitude (central difference)
                double gradX = (field[y * width + (x+1)] - field[y * width + (x-1)]) / 2.0;
                double gradY = (field[(y+1) * width + x] - field[(y-1) * width + x]) / 2.0;
                double gradMag = std::sqrt(gradX*gradX + gradY*gradY);
                
                // Mark for refinement if gradient is above threshold
                refine.set(idx, (gradMag > 0.01));
            }
        }
        
        // Expand refinement to include neighboring cells
        BitArray expandedRefine = refine;
        
        for (size_t idx = refine.findFirst(); idx < refine.size(); idx = refine.findNext(idx)) {
            // Convert linear index to 2D coordinates
            size_t x = idx % width;
            size_t y = idx / width;
            
            if (x > 0 && y > 0 && x < width - 1 && y < height - 1) {
                // Mark neighboring cells
                expandedRefine.set((y-1) * width + x, true);    // North
                expandedRefine.set(y * width + (x+1), true);    // East
                expandedRefine.set((y+1) * width + x, true);    // South
                expandedRefine.set(y * width + (x-1), true);    // West
            }
        }
        
        benchmark::DoNotOptimize(expandedRefine);
    }
    
    state.SetComplexityN(state.range(0));
    state.SetItemsProcessed(state.iterations() * size);
}

/**
 * @brief Benchmark query-based operation with std::vector<bool>
 * 
 * This simulates finding all cells in a region that match a data value criteria.
 */
static void BM_StdVector_DataQuery(benchmark::State& state) {
    size_t size = state.range(0);
    double density = static_cast<double>(state.range(1)) / 100.0;
    
    // Setup: create a region and a data field
    auto indices = createRandomBitPattern(size, density);
    std::vector<bool> region = initializeStdVector(size, indices);
    
    // Create a random data field
    std::vector<double> dataField(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    for (size_t i = 0; i < size; ++i) {
        dataField[i] = dist(gen);
    }
    
    for (auto _ : state) {
        // Find all cells in the region with data values above threshold
        std::vector<size_t> result;
        result.reserve(region.size() / 4);  // Estimate ~25% match
        
        for (size_t i = 0; i < size; ++i) {
            // Check if cell is in region AND data value exceeds threshold
            if (region[i] && dataField[i] > 0.7) {
                result.push_back(i);
            }
        }
        
        benchmark::DoNotOptimize(result);
    }
    
    state.SetComplexityN(state.range(0));
    state.SetItemsProcessed(state.iterations() * size);
}

/**
 * @brief Benchmark region merging operations with std::vector<bool>
 * 
 * This simulates merging multiple material regions in a simulation.
 */
static void BM_StdVector_RegionMerging(benchmark::State& state) {
    size_t size = state.range(0);
    double density = static_cast<double>(state.range(1)) / 100.0;
    
    // Create multiple overlapping regions (simulating materials)
    const int NUM_REGIONS = 5;
    std::vector<std::vector<bool>> regions;
    
    for (int r = 0; r < NUM_REGIONS; ++r) {
        // Create random region with overlap to previous regions
        auto indices = createRandomBitPattern(size, density);
        regions.push_back(initializeStdVector(size, indices));
    }
    
    for (auto _ : state) {
        // Merge regions (material union operation)
        std::vector<bool> merged(size, false);
        
        // Perform union of all regions
        for (size_t i = 0; i < size; ++i) {
            for (const auto& region : regions) {
                if (region[i]) {
                    merged[i] = true;
                    break;  // Once a cell is in any region, it's in the union
                }
            }
        }
        
        // Calculate size of merged region
        size_t mergedSize = 0;
        for (bool bit : merged) {
            if (bit) mergedSize++;
        }
        
        benchmark::DoNotOptimize(merged);
        benchmark::DoNotOptimize(mergedSize);
    }
    
    state.SetComplexityN(state.range(0));
    state.SetItemsProcessed(state.iterations() * size * NUM_REGIONS);
}

/**
 * @brief Benchmark parallel region operations with BitArray
 * 
 * This simulates parallel processing of region data, such as during
 * multi-threaded simulation updates.
 */
static void BM_BitArray_ParallelRegionProcessing(benchmark::State& state) {
    // Only run this benchmark if size is large enough for parallel processing
    if (state.range(0) < 100000) {
        state.SkipWithError("Skipping small size for parallel benchmark");
        return;
    }
    
    size_t size = state.range(0);
    double density = static_cast<double>(state.range(1)) / 100.0;
    
    // Setup: create a region
    auto indices = createRandomBitPattern(size, density);
    BitArray region = initializeBitArray(size, indices);
    
    // Create a data field to update
    std::vector<double> dataField(size, 0.0);
    
    for (auto _ : state) {
        // Process all cells in the region in parallel
        std::vector<size_t> regionIndices;
        regionIndices.reserve(indices.size());
        
        // Gather indices of cells in the region using efficient iteration
        for (size_t idx = region.findFirst(); idx < region.size(); idx = region.findNext(idx)) {
            regionIndices.push_back(idx);
        }
        
        // Now process these cells in parallel
        std::for_each(
            std::execution::par_unseq,
            regionIndices.begin(), regionIndices.end(),
            [&](size_t idx) {
                // Simulate some computation on each cell in the region
                dataField[idx] = std::sin(static_cast<double>(idx) * 0.01) + 
                                 std::cos(static_cast<double>(idx) * 0.05);
            }
        );
        
        benchmark::DoNotOptimize(dataField);
    }
    
    state.SetComplexityN(state.range(0));
    state.SetItemsProcessed(state.iterations() * indices.size());
}

/**
 * @brief Benchmark parallel region operations with std::vector<bool>
 * 
 * This simulates parallel processing of region data, such as during
 * multi-threaded simulation updates.
 */
static void BM_StdVector_ParallelRegionProcessing(benchmark::State& state) {
    // Only run this benchmark if size is large enough for parallel processing
    if (state.range(0) < 100000) {
        state.SkipWithError("Skipping small size for parallel benchmark");
        return;
    }
    
    size_t size = state.range(0);
    double density = static_cast<double>(state.range(1)) / 100.0;
    
    // Setup: create a region
    auto indices = createRandomBitPattern(size, density);
    std::vector<bool> region = initializeStdVector(size, indices);
    
    // Create a data field to update
    std::vector<double> dataField(size, 0.0);
    
    for (auto _ : state) {
        // Process all cells in the region in parallel
        std::vector<size_t> regionIndices;
        regionIndices.reserve(indices.size());
        
        // First, gather indices of cells in the region (cannot directly iterate std::vector<bool>)
        for (size_t i = 0; i < size; ++i) {
            if (region[i]) {
                regionIndices.push_back(i);
            }
        }
        
        // Now process these cells in parallel
        std::for_each(
            std::execution::par_unseq,
            regionIndices.begin(), regionIndices.end(),
            [&](size_t idx) {
                // Simulate some computation on each cell in the region
                dataField[idx] = std::sin(static_cast<double>(idx) * 0.01) + 
                                 std::cos(static_cast<double>(idx) * 0.05);
            }
        );
        
        benchmark::DoNotOptimize(dataField);
    }
    
    state.SetComplexityN(state.range(0));
    state.SetItemsProcessed(state.iterations() * indices.size());
}

//----------------------------------------------------------
// Benchmark Registration
//----------------------------------------------------------

// Basic operations benchmarks with various sizes and densities
static void CustomArguments(benchmark::internal::Benchmark* b) {
    // Small mesh, varying densities
    b->Args({SMALL_MESH_SIZE, static_cast<int>(SPARSE_DENSITY * 100)});    // Small mesh, sparse
    b->Args({SMALL_MESH_SIZE, static_cast<int>(MEDIUM_DENSITY * 100)});    // Small mesh, medium
    b->Args({SMALL_MESH_SIZE, static_cast<int>(HIGH_DENSITY * 100)});      // Small mesh, dense
    
    // Medium mesh, varying densities
    b->Args({MEDIUM_MESH_SIZE, static_cast<int>(SPARSE_DENSITY * 100)});   // Medium mesh, sparse
    b->Args({MEDIUM_MESH_SIZE, static_cast<int>(MEDIUM_DENSITY * 100)});   // Medium mesh, medium
    
    // Large mesh, sparse density only (to keep run time reasonable)
    b->Args({LARGE_MESH_SIZE, static_cast<int>(SPARSE_DENSITY * 100)});    // Large mesh, sparse
}

// Register basic operations benchmarks
BENCHMARK(BM_StdVector_SetBits)->Apply(CustomArguments);
BENCHMARK(BM_BitArray_SetBits)->Apply(CustomArguments);
BENCHMARK(BM_StdVector_GetBits)->Apply(CustomArguments);
BENCHMARK(BM_BitArray_GetBits)->Apply(CustomArguments);
BENCHMARK(BM_StdVector_CountBits)->Apply(CustomArguments);
BENCHMARK(BM_BitArray_CountBits)->Apply(CustomArguments);

// Register set operations benchmarks
BENCHMARK(BM_StdVector_Union)->Apply(CustomArguments);
BENCHMARK(BM_BitArray_Union)->Apply(CustomArguments);
BENCHMARK(BM_StdVector_Intersection)->Apply(CustomArguments);
BENCHMARK(BM_BitArray_Intersection)->Apply(CustomArguments);
BENCHMARK(BM_StdVector_Difference)->Apply(CustomArguments);
BENCHMARK(BM_BitArray_Difference)->Apply(CustomArguments);

// Register iteration benchmarks
BENCHMARK(BM_StdVector_FindBits)->Apply(CustomArguments);
BENCHMARK(BM_BitArray_FindBits)->Apply(CustomArguments);

// Register computational workload benchmarks
BENCHMARK(BM_StdVector_BoundaryExtraction)->Apply(CustomArguments);
BENCHMARK(BM_BitArray_BoundaryExtraction)->Apply(CustomArguments);
BENCHMARK(BM_StdVector_AdaptiveRefinement)->Apply(CustomArguments);
BENCHMARK(BM_BitArray_AdaptiveRefinement)->Apply(CustomArguments);
BENCHMARK(BM_StdVector_DataQuery)->Apply(CustomArguments);
BENCHMARK(BM_BitArray_DataQuery)->Apply(CustomArguments);
BENCHMARK(BM_StdVector_RegionMerging)->Apply(CustomArguments);
BENCHMARK(BM_BitArray_RegionMerging)->Apply(CustomArguments);

// Register parallel benchmarks (only for medium and large sizes)
static void ParallelArguments(benchmark::internal::Benchmark* b) {
    b->Args({MEDIUM_MESH_SIZE, static_cast<int>(MEDIUM_DENSITY * 100)});   // Medium mesh, medium
    b->Args({LARGE_MESH_SIZE, static_cast<int>(SPARSE_DENSITY * 100)});    // Large mesh, sparse
}

BENCHMARK(BM_StdVector_ParallelRegionProcessing)->Apply(ParallelArguments);
BENCHMARK(BM_BitArray_ParallelRegionProcessing)->Apply(ParallelArguments);

// Run the benchmark
BENCHMARK_MAIN();


