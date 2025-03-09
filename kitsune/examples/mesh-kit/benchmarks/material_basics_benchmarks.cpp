/**
 * ====================================================================
 * Material Performance Benchmark Tests
 * ====================================================================
 *
 * These benchmarks measure the performance of key operations in the
 * Material class, including property lookups, mixing operations, and
 * temperature-dependent property calculations.
 *
 * Benchmark Categories:
 * ---------------------
 * 1. Property Access:
 *    - Base property access
 *    - Temperature-dependent property calculations
 *
 * 2. Material Creation:
 *    - Creating predefined materials
 *    - Creating custom materials
 *
 * 3. Mixing Operations:
 *    - Binary mixing
 *    - Multi-component mixing
 *    - Different mixing rules
 *
 * 4. Temperature Models:
 *    - Different property models (constant, linear, exponential, etc.)
 */

#include "Material.h"
#include "Units.h"
#include <benchmark/benchmark.h>
#include <memory>
#include <vector>

// ====================================================================
// Property Access Benchmarks
// ====================================================================

// Benchmark basic property access
static void BM_PropertyAccess(benchmark::State &state) {
  auto water = Material::createPredefined("water");

  for (auto _ : state) {
    double density = water->getProperty(Material::MaterialProperty::DENSITY);
    double viscosity =
        water->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY);
    double conductivity =
        water->getProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY);
    double specificHeat =
        water->getProperty(Material::MaterialProperty::SPECIFIC_HEAT);

    // Prevent compiler optimization from eliminating the calculations
    benchmark::DoNotOptimize(density);
    benchmark::DoNotOptimize(viscosity);
    benchmark::DoNotOptimize(conductivity);
    benchmark::DoNotOptimize(specificHeat);
  }
}
BENCHMARK(BM_PropertyAccess);

// Benchmark temperature-dependent property calculation
static void BM_TemperatureDependentProperty(benchmark::State &state) {
  auto water = Material::createPredefined("water");

  // Generate a set of temperatures to evaluate
  std::vector<double> temperatures;
  for (int i = 0; i < 10; ++i) {
    temperatures.push_back(273.15 + i * 10.0); // 0°C to 90°C in steps of 10°C
  }

  size_t index = 0;
  for (auto _ : state) {
    double temp = temperatures[index % temperatures.size()];
    double density = water->getPropertyAtTemperature(
        Material::MaterialProperty::DENSITY, temp);
    double viscosity = water->getPropertyAtTemperature(
        Material::MaterialProperty::DYNAMIC_VISCOSITY, temp);

    benchmark::DoNotOptimize(density);
    benchmark::DoNotOptimize(viscosity);
    index++;
  }
}
BENCHMARK(BM_TemperatureDependentProperty);

// ====================================================================
// Material Creation Benchmarks
// ====================================================================

// Benchmark creating predefined materials
static void BM_CreatePredefinedMaterial(benchmark::State &state) {
  const char *materialNames[] = {"water",  "air",   "aluminum",
                                 "copper", "steel", "oil"};

  size_t index = 0;
  for (auto _ : state) {
    auto material = Material::createPredefined(materialNames[index % 6]);
    benchmark::DoNotOptimize(material);
    index++;
  }
}
BENCHMARK(BM_CreatePredefinedMaterial);

// Benchmark creating custom materials
static void BM_CreateCustomMaterial(benchmark::State &state) {
  for (auto _ : state) {
    auto material = std::make_shared<Material>(Material::MaterialType::FLUID,
                                               "CustomMaterial");

    material->setProperty(Material::MaterialProperty::DENSITY, 850.0);
    material->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY, 0.05);
    material->setProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY,
                          0.12);
    material->setProperty(Material::MaterialProperty::SPECIFIC_HEAT, 2000.0);

    benchmark::DoNotOptimize(material);
  }
}
BENCHMARK(BM_CreateCustomMaterial);

// ====================================================================
// Mixing Operation Benchmarks
// ====================================================================

// Benchmark binary mixing
static void BM_BinaryMixing(benchmark::State &state) {
  auto water = Material::createPredefined("water");
  auto air = Material::createPredefined("air");

  std::vector<double> fractions = {0.0, 0.2, 0.4, 0.6, 0.8, 1.0};

  size_t index = 0;
  for (auto _ : state) {
    double fraction = fractions[index % fractions.size()];
    auto mixture = water->createMixture(air, fraction);
    benchmark::DoNotOptimize(mixture);
    index++;
  }
}
BENCHMARK(BM_BinaryMixing);

// Benchmark multi-component mixing
static void BM_MultiComponentMixing(benchmark::State &state) {
  // Number of components to mix
  int numComponents = state.range(0);

  // Create materials for mixing
  std::vector<std::shared_ptr<Material>> materials;
  std::vector<double> fractions;

  // Materials will be a combination of predefined materials
  auto water = Material::createPredefined("water");
  auto air = Material::createPredefined("air");
  auto oil = Material::createPredefined("oil");

  // Create as many materials as needed
  for (int i = 0; i < numComponents; ++i) {
    switch (i % 3) {
    case 0:
      materials.push_back(water);
      break;
    case 1:
      materials.push_back(air);
      break;
    case 2:
      materials.push_back(oil);
      break;
    }
    fractions.push_back(1.0 / numComponents);
  }

  for (auto _ : state) {
    auto mixture = Material::createMixture(materials, fractions);
    benchmark::DoNotOptimize(mixture);
  }
}
// Test with different numbers of components
BENCHMARK(BM_MultiComponentMixing)->Arg(2)->Arg(4)->Arg(8)->Arg(16);

// Benchmark different mixing rules
static void BM_MixingRules(benchmark::State &state) {
  auto water = Material::createPredefined("water");
  auto oil = Material::createPredefined("oil");

  // Use an enum for the mixing rule
  Material::MixingRuleType mixingRule;
  switch (state.range(0)) {
  case 0:
    mixingRule = Material::MixingRuleType::LINEAR;
    break;
  case 1:
    mixingRule = Material::MixingRuleType::LOGARITHMIC;
    break;
  case 2:
    mixingRule = Material::MixingRuleType::HARMONIC;
    break;
  case 3:
    mixingRule = Material::MixingRuleType::GEOMETRIC;
    break;
  default:
    mixingRule = Material::MixingRuleType::DEFAULT;
    break;
  }

  for (auto _ : state) {
    auto mixture = water->createMixture(oil, 0.3, mixingRule);
    // Get a few property values to ensure mixing is actually performed
    double density = mixture->getProperty(Material::MaterialProperty::DENSITY);
    double viscosity =
        mixture->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY);

    benchmark::DoNotOptimize(mixture);
    benchmark::DoNotOptimize(density);
    benchmark::DoNotOptimize(viscosity);
  }
}
// Test with different mixing rules
BENCHMARK(BM_MixingRules)->Arg(0)->Arg(1)->Arg(2)->Arg(3)->Arg(4);

// ====================================================================
// Temperature Model Benchmarks
// ====================================================================

// Benchmark different property temperature models
static void BM_PropertyModels(benchmark::State &state) {
  // Create a custom material
  auto material = std::make_shared<Material>(Material::MaterialType::FLUID,
                                             "ModelTestMaterial");
  material->setProperty(Material::MaterialProperty::DENSITY, 1000.0);
  material->setReferenceTemperature(293.15);

  // Set up the temperature model based on benchmark argument
  Material::PropertyModel model;
  switch (state.range(0)) {
  case 0:
    model = Material::PropertyModel::CONSTANT;
    break;
  case 1:
    model = Material::PropertyModel::LINEAR;
    material->setPropertyModel(Material::MaterialProperty::DENSITY, model,
                               {0.001});
    break;
  case 2:
    model = Material::PropertyModel::POLYNOMIAL;
    material->setPropertyModel(Material::MaterialProperty::DENSITY, model,
                               {0.001, -0.000001});
    break;
  case 3:
    model = Material::PropertyModel::EXPONENTIAL;
    material->setPropertyModel(Material::MaterialProperty::DENSITY, model,
                               {-0.001});
    break;
  case 4:
    model = Material::PropertyModel::CUSTOM;
    material->setCustomPropertyFunction(
        Material::MaterialProperty::DENSITY, [](double T) -> double {
          double T_C = T - 273.15;
          return 1000.0 - 0.1 * (T_C - 4.0) * (T_C - 4.0);
        });
    break;
  default:
    model = Material::PropertyModel::CONSTANT;
    break;
  }

  material->setUseTempDependentProps(true);

  // Generate a set of temperatures to evaluate
  std::vector<double> temperatures;
  for (int i = 0; i < 10; ++i) {
    temperatures.push_back(273.15 + i * 10.0);
  }

  size_t index = 0;
  for (auto _ : state) {
    double temp = temperatures[index % temperatures.size()];
    double density = material->getPropertyAtTemperature(
        Material::MaterialProperty::DENSITY, temp);

    benchmark::DoNotOptimize(density);
    index++;
  }
}
// Test with different property models
BENCHMARK(BM_PropertyModels)->Arg(0)->Arg(1)->Arg(2)->Arg(3)->Arg(4);

// ====================================================================
// Real-world Use Case Benchmarks
// ====================================================================

// Benchmark a realistic CFD material lookup scenario
static void BM_CFDMaterialLookup(benchmark::State &state) {
  // Create materials that might be used in a CFD simulation
  auto water = Material::createPredefined("water");
  auto oil = Material::createPredefined("oil");
  auto mixture = water->createMixture(oil, 0.3);

  // Simulate iterating through a grid with varying temperatures
  const int gridSize = state.range(0);
  std::vector<double> temperatures;

  // Create a temperature field (simplified)
  for (int i = 0; i < gridSize; ++i) {
    temperatures.push_back(293.15 + (i % 50)); // Vary by 50K max
  }

  for (auto _ : state) {
    double totalDensity = 0.0;
    double totalViscosity = 0.0;

    for (int i = 0; i < gridSize; ++i) {
      double temp = temperatures[i];
      double density = mixture->getPropertyAtTemperature(
          Material::MaterialProperty::DENSITY, temp);
      double viscosity = mixture->getPropertyAtTemperature(
          Material::MaterialProperty::DYNAMIC_VISCOSITY, temp);

      totalDensity += density;
      totalViscosity += viscosity;
    }

    benchmark::DoNotOptimize(totalDensity);
    benchmark::DoNotOptimize(totalViscosity);
  }
}
// Test with different grid sizes
BENCHMARK(BM_CFDMaterialLookup)->Arg(10)->Arg(100)->Arg(1000);

// ====================================================================
// Unit Conversion Benchmarks
// ====================================================================

// Benchmark unit conversion operations
static void BM_UnitConversion(benchmark::State &state) {
  auto water = Material::createPredefined("water");

  for (auto _ : state) {
    // Get properties in different units
    double density_lbft3 = water->getPropertyWithUnits(
        Material::MaterialProperty::DENSITY, "lb/ft³");
    double visc_centipoise = water->getPropertyWithUnits(
        Material::MaterialProperty::DYNAMIC_VISCOSITY, "cP");
    double cond_btu = water->getPropertyWithUnits(
        Material::MaterialProperty::THERMAL_CONDUCTIVITY, "BTU/(hr·ft·°F)");
    double cp_btu = water->getPropertyWithUnits(
        Material::MaterialProperty::SPECIFIC_HEAT, "BTU/(lb·°F)");

    benchmark::DoNotOptimize(density_lbft3);
    benchmark::DoNotOptimize(visc_centipoise);
    benchmark::DoNotOptimize(cond_btu);
    benchmark::DoNotOptimize(cp_btu);
  }
}
BENCHMARK(BM_UnitConversion);

// Main function to run the benchmarks
BENCHMARK_MAIN();
