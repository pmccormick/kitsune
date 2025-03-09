/**
 * ====================================================================
 * Material Property Cache Benchmark Tests
 * ====================================================================
 *
 * These benchmarks measure the performance improvements gained from using
 * the MaterialPropertyCache class for temperature-dependent material
 * property lookups.
 *
 * Benchmark Categories:
 * ---------------------
 * 1. Cache Hit/Miss Performance:
 *    - Cold cache vs warm cache performance
 *    - Exact temperature match vs interpolation
 *
 * 2. Cache Size Impact:
 *    - Different access patterns with varying locality
 *    - Impact of LRU policy under different workloads
 *
 * 3. Multiple Property Types:
 *    - Single vs multiple property type lookups
 *    - Cache effectiveness across property types
 *
 * 4. Real-world Simulation Patterns:
 *    - Grid traversal patterns
 *    - Time-stepping temperature changes
 */

#include "Material.h"
#include "MaterialPropertyCache.h"
#include "Units.h"
#include <benchmark/benchmark.h>
#include <memory>
#include <random>
#include <vector>

// Define the property enum equivalent to what's in Material.h for testing
// purposes
enum class TestProperty : size_t {
  DENSITY = 0,
  DYNAMIC_VISCOSITY = 1,
  THERMAL_CONDUCTIVITY = 2,
  SPECIFIC_HEAT = 3,
  COUNT = 4 // Used to specify the number of properties
};

// Helper function to create a temperature-dependent property function
// This simulates expensive property calculations that would benefit from
// caching
double calculatePropertyValue(TestProperty property, double temperature) {
  // Add a small delay to simulate computational cost
  for (volatile int i = 0; i < 100; i++) {
  }

  // Different formulas for different properties (just for testing)
  switch (property) {
  case TestProperty::DENSITY:
    return 1000.0 - 0.1 * (temperature - 293.15);
  case TestProperty::DYNAMIC_VISCOSITY:
    return 0.001 * exp(-0.02 * (temperature - 293.15));
  case TestProperty::THERMAL_CONDUCTIVITY:
    return 0.6 + 0.0002 * (temperature - 293.15);
  case TestProperty::SPECIFIC_HEAT:
    return 4200.0 + 0.1 * (temperature - 293.15);
  default:
    return 0.0;
  }
}

// ====================================================================
// Cache Hit/Miss Performance Benchmarks
// ====================================================================

// Benchmark for cold cache (all misses)
static void BM_ColdCache(benchmark::State &state) {
  MaterialPropertyCache<TestProperty, static_cast<size_t>(TestProperty::COUNT)>
      cache;

  // Generate random temperatures for testing
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> distrib(273.15,
                                           373.15); // Range: 0°C to 100°C

  std::vector<double> temperatures;
  for (int i = 0; i < 1000; i++) {
    temperatures.push_back(distrib(gen));
  }

  // Select property based on benchmark arg
  TestProperty property = static_cast<TestProperty>(state.range(0));

  size_t index = 0;
  for (auto _ : state) {
    double temp = temperatures[index % temperatures.size()];
    double value;

    if (!cache.lookup(property, temp, value)) {
      // Cache miss - calculate and store
      value = calculatePropertyValue(property, temp);
      cache.store(property, temp, value);
    }

    benchmark::DoNotOptimize(value);
    index++;
  }
}
BENCHMARK(BM_ColdCache)
    ->Arg(static_cast<int>(TestProperty::DENSITY))
    ->Arg(static_cast<int>(TestProperty::DYNAMIC_VISCOSITY))
    ->Arg(static_cast<int>(TestProperty::THERMAL_CONDUCTIVITY))
    ->Arg(static_cast<int>(TestProperty::SPECIFIC_HEAT));

// Benchmark for warm cache (mostly hits)
static void BM_WarmCache(benchmark::State &state) {
  MaterialPropertyCache<TestProperty, static_cast<size_t>(TestProperty::COUNT)>
      cache;

  // Use a fixed set of temperatures with high repetition
  const int numTemps = PropertyCache::MAX_CACHE_ENTRIES;
  std::vector<double> temperatures;
  for (int i = 0; i < numTemps; i++) {
    temperatures.push_back(293.15 + i * 10.0); // 20°C to 90°C in steps of 10°C
  }

  // Select property based on benchmark arg
  TestProperty property = static_cast<TestProperty>(state.range(0));

  // Pre-warm the cache
  for (double temp : temperatures) {
    double value = calculatePropertyValue(property, temp);
    cache.store(property, temp, value);
  }

  // Now benchmark with the warm cache
  size_t index = 0;
  for (auto _ : state) {
    double temp = temperatures[index % temperatures.size()];
    double value;

    if (!cache.lookup(property, temp, value)) {
      // Should rarely happen with pre-warmed cache
      value = calculatePropertyValue(property, temp);
      cache.store(property, temp, value);
    }

    benchmark::DoNotOptimize(value);
    index++;
  }
}
BENCHMARK(BM_WarmCache)
    ->Arg(static_cast<int>(TestProperty::DENSITY))
    ->Arg(static_cast<int>(TestProperty::DYNAMIC_VISCOSITY))
    ->Arg(static_cast<int>(TestProperty::THERMAL_CONDUCTIVITY))
    ->Arg(static_cast<int>(TestProperty::SPECIFIC_HEAT));

// Benchmark to compare cached vs non-cached lookup
static void BM_CachedVsNonCached(benchmark::State &state) {
  MaterialPropertyCache<TestProperty, static_cast<size_t>(TestProperty::COUNT)>
      cache;

  // Use temperatures with some locality
  std::vector<double> temperatures;
  for (int i = 0; i < 100; i++) {
    temperatures.push_back(293.15 +
                           (i % 20) * 5.0); // 20°C to 115°C with repetition
  }

  // Select property
  TestProperty property = static_cast<TestProperty>(TestProperty::DENSITY);

  // Choose whether to use cache based on benchmark arg
  bool useCache = state.range(0) == 1;

  size_t index = 0;
  for (auto _ : state) {
    double temp = temperatures[index % temperatures.size()];
    double value;

    if (useCache) {
      // Try cache first
      if (!cache.lookup(property, temp, value)) {
        value = calculatePropertyValue(property, temp);
        cache.store(property, temp, value);
      }
    } else {
      // Always calculate without cache
      value = calculatePropertyValue(property, temp);
    }

    benchmark::DoNotOptimize(value);
    index++;
  }
}
BENCHMARK(BM_CachedVsNonCached)
    ->Arg(0)  // No cache
    ->Arg(1); // With cache

// ====================================================================
// Cache Size Impact Benchmarks
// ====================================================================

// Benchmark different access patterns
static void BM_AccessPatterns(benchmark::State &state) {
  MaterialPropertyCache<TestProperty, static_cast<size_t>(TestProperty::COUNT)>
      cache;

  // Generate temperatures based on the access pattern type
  std::vector<double> temperatures;

  // Pattern type based on benchmark arg:
  // 0: Sequential access (poor for LRU if cache smaller than range)
  // 1: Repeated small set (good locality)
  // 2: Random access with bias (some locality)
  int patternType = state.range(0);

  if (patternType == 0) {
    // Sequential pattern: 0°C to 100°C in 1°C steps
    for (int i = 0; i < 100; i++) {
      temperatures.push_back(273.15 + i);
    }
  } else if (patternType == 1) {
    // Small set with high repetition (good cache utilization)
    for (int i = 0; i < 100; i++) {
      temperatures.push_back(293.15 +
                             (i % 5) * 5.0); // Only 5 distinct temperatures
    }
  } else {
    // Random with bias (some locality)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> distrib(323.15, 10.0); // Mean 50°C, stddev 10°C

    for (int i = 0; i < 100; i++) {
      temperatures.push_back(distrib(gen));
    }
  }

  // Select property
  TestProperty property =
      static_cast<TestProperty>(TestProperty::THERMAL_CONDUCTIVITY);

  size_t index = 0;
  size_t hits = 0, misses = 0;

  for (auto _ : state) {
    double temp = temperatures[index % temperatures.size()];
    double value;

    if (cache.lookup(property, temp, value)) {
      hits++;
    } else {
      value = calculatePropertyValue(property, temp);
      cache.store(property, temp, value);
      misses++;
    }

    benchmark::DoNotOptimize(value);
    index++;
  }

  // Report cache hit rate as a custom counter
  state.counters["HitRate"] = benchmark::Counter(
      static_cast<double>(hits) / static_cast<double>(hits + misses),
      benchmark::Counter::kAvgIterations);
}
BENCHMARK(BM_AccessPatterns)
    ->Arg(0)  // Sequential
    ->Arg(1)  // High locality
    ->Arg(2); // Random with bias

// ====================================================================
// Multiple Property Types Benchmarks
// ====================================================================

// Benchmark for multiple property lookups
static void BM_MultiPropertyLookup(benchmark::State &state) {
  MaterialPropertyCache<TestProperty, static_cast<size_t>(TestProperty::COUNT)>
      cache;

  // Choose number of properties to look up based on benchmark arg
  int numProperties = state.range(0);

  // Fixed set of temperatures
  std::vector<double> temperatures;
  for (int i = 0; i < 20; i++) {
    temperatures.push_back(293.15 + i * 5.0);
  }

  size_t index = 0;
  for (auto _ : state) {
    double temp = temperatures[index % temperatures.size()];
    double totalValue = 0.0;

    // Look up multiple properties
    for (int i = 0; i < numProperties; i++) {
      TestProperty property =
          static_cast<TestProperty>(i % static_cast<int>(TestProperty::COUNT));
      double value;

      if (!cache.lookup(property, temp, value)) {
        value = calculatePropertyValue(property, temp);
        cache.store(property, temp, value);
      }

      totalValue += value;
    }

    benchmark::DoNotOptimize(totalValue);
    index++;
  }
}
BENCHMARK(BM_MultiPropertyLookup)
    ->Arg(1)  // Single property
    ->Arg(2)  // Two properties
    ->Arg(4); // All properties

// ====================================================================
// Real-world Simulation Pattern Benchmarks
// ====================================================================

// Benchmark for grid traversal pattern (common in CFD)
static void BM_GridTraversal(benchmark::State &state) {
  MaterialPropertyCache<TestProperty, static_cast<size_t>(TestProperty::COUNT)>
      cache;

  // Grid size based on benchmark arg
  int gridSize = state.range(0);

  // Create a 2D temperature field with some spatial coherence
  std::vector<std::vector<double>> temperatureGrid(
      gridSize, std::vector<double>(gridSize));

  // Initialize with a simple temperature gradient
  for (int i = 0; i < gridSize; i++) {
    for (int j = 0; j < gridSize; j++) {
      temperatureGrid[i][j] = 293.15 + 0.1 * (i + j);
    }
  }

  // Choose a property to look up
  TestProperty property = TestProperty::DENSITY;

  // Choose traversal order based on benchmark arg
  int traversalOrder = state.range(1);

  for (auto _ : state) {
    double totalValue = 0.0;

    if (traversalOrder == 0) {
      // Row-major traversal (good spatial locality)
      for (int i = 0; i < gridSize; i++) {
        for (int j = 0; j < gridSize; j++) {
          double temp = temperatureGrid[i][j];
          double value;

          if (!cache.lookup(property, temp, value)) {
            value = calculatePropertyValue(property, temp);
            cache.store(property, temp, value);
          }

          totalValue += value;
        }
      }
    } else {
      // Column-major traversal (worse for temperature locality)
      for (int j = 0; j < gridSize; j++) {
        for (int i = 0; i < gridSize; i++) {
          double temp = temperatureGrid[i][j];
          double value;

          if (!cache.lookup(property, temp, value)) {
            value = calculatePropertyValue(property, temp);
            cache.store(property, temp, value);
          }

          totalValue += value;
        }
      }
    }

    benchmark::DoNotOptimize(totalValue);
  }
}
BENCHMARK(BM_GridTraversal)
    ->Args({10, 0})  // 10x10 grid, row-major
    ->Args({10, 1})  // 10x10 grid, column-major
    ->Args({20, 0})  // 20x20 grid, row-major
    ->Args({20, 1}); // 20x20 grid, column-major

// Benchmark for time-stepping behavior (temperature changes over time)
static void BM_Timestepping(benchmark::State &state) {
  MaterialPropertyCache<TestProperty, static_cast<size_t>(TestProperty::COUNT)>
      cache;

  // Number of points in our domain
  int numPoints = state.range(0);

  // Number of time steps
  int numTimeSteps = 10;

  // Initial temperatures for each point
  std::vector<double> initialTemperatures(numPoints);
  for (int i = 0; i < numPoints; i++) {
    initialTemperatures[i] = 293.15 + i * 10.0 / numPoints;
  }

  // Choose a property
  TestProperty property = TestProperty::DYNAMIC_VISCOSITY;

  // Time step size (in terms of temperature change)
  double deltaT = state.range(1) * 0.1; // Temperature change per time step

  for (auto _ : state) {
    // Reset cache at the beginning of each measurement
    cache.clear();

    double totalValue = 0.0;

    // Time stepping loop
    for (int step = 0; step < numTimeSteps; step++) {
      // Domain traversal loop
      for (int i = 0; i < numPoints; i++) {
        // Calculate current temperature with time evolution
        double temp = initialTemperatures[i] + step * deltaT;
        double value;

        if (!cache.lookup(property, temp, value)) {
          value = calculatePropertyValue(property, temp);
          cache.store(property, temp, value);
        }

        totalValue += value;
      }
    }

    benchmark::DoNotOptimize(totalValue);
  }
}
BENCHMARK(BM_Timestepping)
    ->Args({100, 1})   // 100 points, small temperature change
    ->Args({100, 5})   // 100 points, medium temperature change
    ->Args({100, 10}); // 100 points, large temperature change

// ====================================================================
// Cache Boundary Condition Benchmarks
// ====================================================================

// Benchmark behavior around cache entry precision boundaries
static void BM_CachePrecisionBoundary(benchmark::State &state) {
  MaterialPropertyCache<TestProperty, static_cast<size_t>(TestProperty::COUNT)>
      cache;

  // Choose a property
  TestProperty property = TestProperty::SPECIFIC_HEAT;

  // Choose precision offset as a fraction of TEMPERATURE_PRECISION
  double precisionFactor = state.range(0) * 0.1;

  // Base temperature
  double baseTemp = 300.0;

  // Precision boundary
  double precision = PropertyCache::TEMPERATURE_PRECISION;

  // Create slightly different temperatures around precision boundary
  double temp1 = baseTemp;
  double temp2 = baseTemp + precision * precisionFactor;

  // Pre-warm cache with the first temperature
  double value1 = calculatePropertyValue(property, temp1);
  cache.store(property, temp1, value1);

  for (auto _ : state) {
    // Look up the second temperature (might be within precision or not)
    double value;
    bool hit = cache.lookup(property, temp2, value);

    if (!hit) {
      value = calculatePropertyValue(property, temp2);
      cache.store(property, temp2, value);
    }

    benchmark::DoNotOptimize(value);
    benchmark::DoNotOptimize(hit);
  }

  // Report whether this was a hit or miss scenario
  state.SetLabel(std::string("DistanceFactor: ") +
                 std::to_string(precisionFactor) +
                 ", Hit: " + (precisionFactor < 1.0 ? "true" : "false"));
}
BENCHMARK(BM_CachePrecisionBoundary)
    ->Arg(0)   // Same temperature
    ->Arg(5)   // Half precision distance
    ->Arg(9)   // Just under precision boundary
    ->Arg(10)  // Right at precision boundary
    ->Arg(11); // Just over precision boundary

// ====================================================================
// Cache Overflow Handling Benchmarks
// ====================================================================

// Benchmark for access counter overflow handling
static void BM_CounterOverflowHandling(benchmark::State &state) {
  MaterialPropertyCache<TestProperty, static_cast<size_t>(TestProperty::COUNT)>
      cache;

  // Choose a property
  TestProperty property = TestProperty::DENSITY;

  // Number of accesses before forcing an overflow
  const uint32_t overflowThreshold = std::numeric_limits<uint32_t>::max() -
                                     PropertyCache::MAX_CACHE_ENTRIES - 10;

  // Generate some temperatures
  std::vector<double> temperatures;
  for (unsigned i = 0; i < PropertyCache::MAX_CACHE_ENTRIES; i++) {
    temperatures.push_back(293.15 + i * 10.0);
  }

  // Pre-warm cache and artificially advance the access counter
  for (size_t i = 0; i < temperatures.size(); i++) {
    double value = calculatePropertyValue(property, temperatures[i]);
    cache.store(property, temperatures[i], value);

    // Only advance counter if specified by benchmark arg
    if (state.range(0) == 1) {
      // Simulate many accesses to force overflow
      for (uint32_t j = 0;
           j < overflowThreshold / PropertyCache::MAX_CACHE_ENTRIES; j++) {
        double dummy;
        cache.lookup(property, temperatures[i], dummy);
      }
    }
  }

  // Force counter overflow handling if needed
  if (state.range(0) == 1) {
    cache.handleCounterOverflow();
  }

  // Now benchmark lookups after potential overflow handling
  size_t index = 0;
  for (auto _ : state) {
    double temp = temperatures[index % temperatures.size()];
    double value;

    bool hit = cache.lookup(property, temp, value);
    if (!hit) {
      value = calculatePropertyValue(property, temp);
      cache.store(property, temp, value);
    }

    benchmark::DoNotOptimize(value);
    index++;
  }
}
BENCHMARK(BM_CounterOverflowHandling)
    ->Arg(0)  // Normal operation
    ->Arg(1); // After overflow handling

// ====================================================================
// Material Integration Benchmarks
// ====================================================================

// Benchmark for integrating the cache with actual Material class usage
// This requires proper integration with the Material class
// Here we simulate the integration by focusing on the caching logic
static void BM_MaterialClassIntegration(benchmark::State &state) {
  MaterialPropertyCache<TestProperty, static_cast<size_t>(TestProperty::COUNT)>
      cache;

  // Simulate material properties
  std::vector<double> temperatures;
  for (int i = 0; i < 20; i++) {
    temperatures.push_back(293.15 + i * 5.0);
  }

  // Use cache based on benchmark arg
  bool useCache = state.range(0) == 1;

  size_t index = 0;
  for (auto _ : state) {
    // Simulate a typical usage pattern in Material::getPropertyAtTemperature
    double totalValue = 0.0;

    // Look up multiple properties at the same temperature
    double temp = temperatures[index % temperatures.size()];

    for (int i = 0; i < static_cast<int>(TestProperty::COUNT); i++) {
      TestProperty property = static_cast<TestProperty>(i);
      double value;

      if (useCache) {
        if (!cache.lookup(property, temp, value)) {
          value = calculatePropertyValue(property, temp);
          cache.store(property, temp, value);
        }
      } else {
        value = calculatePropertyValue(property, temp);
      }

      totalValue += value;
    }

    benchmark::DoNotOptimize(totalValue);
    index++;
  }
}
BENCHMARK(BM_MaterialClassIntegration)
    ->Arg(0)  // Without cache
    ->Arg(1); // With cache

// ====================================================================
// Mixed Access Pattern Benchmarks
// ====================================================================

// Benchmark more realistic mixed access patterns
static void BM_MixedAccessPattern(benchmark::State &state) {
  MaterialPropertyCache<TestProperty, static_cast<size_t>(TestProperty::COUNT)>
      cache;

  // Generate temperature patterns based on the scenario
  std::vector<double> temperatures;

  // Different use case scenarios:
  // 0: CFD-like pattern: Small oscillations around a central value
  // 1: Heat transfer: Gradual but consistent temperature rise
  // 2: Mixing: Alternating between two temperature regions
  int scenario = state.range(0);

  std::random_device rd;
  std::mt19937 gen(rd());

  if (scenario == 0) {
    // CFD-like pattern
    std::normal_distribution<> distrib(323.15,
                                       3.0); // Mean 50°C, small variation
    for (int i = 0; i < 100; i++) {
      temperatures.push_back(distrib(gen));
    }
  } else if (scenario == 1) {
    // Heat transfer
    double baseTemp = 293.15;
    for (int i = 0; i < 100; i++) {
      temperatures.push_back(baseTemp + i * 0.5); // Gradual increase
    }
  } else {
    // Mixing simulation
    std::uniform_real_distribution<> coldDist(283.15, 293.15); // Cold region
    std::uniform_real_distribution<> hotDist(353.15, 363.15);  // Hot region
    for (int i = 0; i < 100; i++) {
      if (i % 2 == 0) {
        temperatures.push_back(coldDist(gen));
      } else {
        temperatures.push_back(hotDist(gen));
      }
    }
  }

  // Do we use the cache?
  bool useCache = state.range(1) == 1;

  // Properties to look up
  std::vector<TestProperty> properties = {TestProperty::DENSITY,
                                          TestProperty::DYNAMIC_VISCOSITY,
                                          TestProperty::THERMAL_CONDUCTIVITY};

  size_t index = 0;
  for (auto _ : state) {
    double temp = temperatures[index % temperatures.size()];
    TestProperty property = properties[index % properties.size()];
    double value;

    if (useCache) {
      if (!cache.lookup(property, temp, value)) {
        value = calculatePropertyValue(property, temp);
        cache.store(property, temp, value);
      }
    } else {
      value = calculatePropertyValue(property, temp);
    }

    benchmark::DoNotOptimize(value);
    index++;
  }
}
BENCHMARK(BM_MixedAccessPattern)
    ->Args({0, 0})  // CFD pattern, no cache
    ->Args({0, 1})  // CFD pattern, with cache
    ->Args({1, 0})  // Heat transfer, no cache
    ->Args({1, 1})  // Heat transfer, with cache
    ->Args({2, 0})  // Mixing, no cache
    ->Args({2, 1}); // Mixing, with cache

