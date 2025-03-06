#pragma once

#include "Cell.h"
#include "Material.h"
#include <array>
#include <functional>
#include <gmock/gmock.h>
#include <memory>
#include <vector>

/**
 * @brief Creates a cell with predefined material and properties
 *
 * @param cellType Type of cell to create
 * @param material Material to assign to the cell
 * @param temperature Temperature to set (in Kelvin)
 * @param pressure Pressure to set (in Pascal)
 * @param velocityU X-velocity component (in m/s)
 * @param velocityV Y-velocity component (in m/s)
 * @return Unique pointer to the created cell
 */
inline std::unique_ptr<Cell>
createTestCell(Cell::CellType cellType = Cell::CellType::FLUID,
               std::shared_ptr<Material> material = nullptr,
               double temperature = 293.15, double pressure = 101325.0,
               double velocityU = 0.0, double velocityV = 0.0) {
  auto cell = std::make_unique<Cell>(cellType);

  if (material) {
    cell->setMaterial(material);
  }

  cell->setTemperature(temperature);
  cell->setPressure(pressure);
  cell->setVelocityU(velocityU);
  cell->setVelocityV(velocityV);

  return cell;
}

/**
 * @brief Creates a cell with properties specified in non-SI units
 *
 * @param cellType Type of cell to create
 * @param material Material to assign to the cell
 * @param temperature Temperature to set
 * @param tempUnit Temperature unit (e.g., "C", "F")
 * @param pressure Pressure to set
 * @param pressureUnit Pressure unit (e.g., "bar", "atm")
 * @param velocity Velocity magnitude
 * @param velocityUnit Velocity unit (e.g., "mph", "knot")
 * @param angle Flow direction angle in degrees (0 = positive x)
 * @return Unique pointer to the created cell
 */
inline std::unique_ptr<Cell> createTestCellWithUnits(
    Cell::CellType cellType = Cell::CellType::FLUID,
    std::shared_ptr<Material> material = nullptr, double temperature = 20.0,
    const std::string &tempUnit = "C", double pressure = 1.0,
    const std::string &pressureUnit = "bar", double velocity = 0.0,
    const std::string &velocityUnit = "m/s", double angle = 0.0) {
  auto cell = std::make_unique<Cell>(cellType);

  if (material) {
    cell->setMaterial(material);
  }

  // Set cell properties with unit conversion
  cell->setTemperatureWithUnits(temperature, tempUnit);
  cell->setPressureWithUnits(pressure, pressureUnit);

  // Calculate velocity components based on magnitude and angle
  double rad = angle * M_PI / 180.0;
  double vx = velocity * cos(rad);
  double vy = velocity * sin(rad);
  cell->setVelocityWithUnits(vx, vy, velocityUnit);

  return cell;
}

/**
 * @brief Creates a set of neighboring cells for property computation tests
 *
 * @param centerCell Reference to the central cell
 * @param withMaterials Whether to assign materials to cells
 * @param baseTemp Base temperature for all cells
 * @param gradientX Temperature gradient in x-direction (K/m)
 * @param gradientY Temperature gradient in y-direction (K/m)
 * @return Array of Cell pointers in order [North, East, South, West]
 */
inline std::array<Cell *, 4> createNeighborCellsForPropertyTests(
    Cell &centerCell, bool withMaterials = true, double baseTemp = 293.15,
    double gradientX = 10.0, double gradientY = 5.0) {
  static std::vector<std::unique_ptr<Cell>> cells;
  cells.clear();

  // Create four neighboring cells
  for (int i = 0; i < 4; i++) {
    cells.push_back(std::make_unique<Cell>(Cell::CellType::FLUID));

    // Use the same material as center cell if requested
    if (withMaterials && centerCell.getMaterial()) {
      cells[i]->setMaterial(centerCell.getMaterial());
    }
  }

  // Set temperatures with gradients (assume 1m grid spacing for simplicity)
  // North [0]: (0, +1)
  cells[0]->setTemperature(baseTemp + gradientY);

  // East [1]: (+1, 0)
  cells[1]->setTemperature(baseTemp + gradientX);

  // South [2]: (0, -1)
  cells[2]->setTemperature(baseTemp - gradientY);

  // West [3]: (-1, 0)
  cells[3]->setTemperature(baseTemp - gradientX);

  // Create and return the array of pointers
  std::array<Cell *, 4> neighbors = {cells[0].get(), cells[1].get(),
                                     cells[2].get(), cells[3].get()};

  return neighbors;
}

/**
 * @brief Register a test property compute function for testing
 *
 * This function allows registering a custom property compute function
 * for testing property computation on cells.
 *
 * @param propType Property type to compute
 * @param fn Function to compute the property
 */
inline void registerTestPropertyComputation(
    Cell::PropertyType propType,
    std::function<void(Cell &, const std::array<Cell *, 4> *)> fn) {
  Cell::registerPropertyComputation(propType, fn);
}

/**
 * @brief Helper for testing material mixing within cells
 *
 * @param baseCell Cell with initial material
 * @param mixMaterial Material to mix with
 * @param mixFraction Fraction of mixing material to add
 * @param rule Mixing rule to use
 * @return Resulting mixed material properties
 */
inline std::unordered_map<std::string, double>
testCellMaterialMixing(Cell &baseCell, std::shared_ptr<Material> mixMaterial,
                       double mixFraction,
                       const std::string &rule = "default") {
  // Store initial properties
  std::unordered_map<std::string, double> initialProps;
  std::unordered_map<std::string, double> resultProps;

  // Record properties before mixing
  initialProps["density"] = baseCell.getEffectiveDensity();
  initialProps["temperature"] = baseCell.getTemperature();

  if (baseCell.getMaterial()) {
    initialProps["viscosity"] =
        baseCell.getMaterial()->getPropertyAtTemperature(
            Material::MaterialProperty::DYNAMIC_VISCOSITY,
            baseCell.getTemperature());
    initialProps["conductivity"] =
        baseCell.getMaterial()->getPropertyAtTemperature(
            Material::MaterialProperty::THERMAL_CONDUCTIVITY,
            baseCell.getTemperature());
  }

  // Perform the material mixing
  baseCell.mixMaterial(mixMaterial, mixFraction, rule);

  // Record properties after mixing
  resultProps["density"] = baseCell.getEffectiveDensity();
  resultProps["temperature"] = baseCell.getTemperature();

  if (baseCell.getMaterial()) {
    resultProps["viscosity"] = baseCell.getMaterial()->getPropertyAtTemperature(
        Material::MaterialProperty::DYNAMIC_VISCOSITY,
        baseCell.getTemperature());
    resultProps["conductivity"] =
        baseCell.getMaterial()->getPropertyAtTemperature(
            Material::MaterialProperty::THERMAL_CONDUCTIVITY,
            baseCell.getTemperature());
  }

  return resultProps;
}

/**
 * @brief Test utility to compute cell properties for all neighbor
 * configurations
 *
 * This function computes properties with different combinations of neighboring
 * cells to test robustness of property computation functions.
 *
 * @param cell Cell to test property computation on
 * @param propType Property type to compute
 * @return Map of results for different neighbor configurations
 */
inline std::unordered_map<std::string, double>
testPropertyComputationRobustness(Cell &cell, Cell::PropertyType propType) {
  std::unordered_map<std::string, double> results;

  // Create temporary neighboring cells
  std::array<std::unique_ptr<Cell>, 4> tempCells;
  for (int i = 0; i < 4; i++) {
    tempCells[i] = std::make_unique<Cell>(Cell::CellType::FLUID);
    // Set some reasonable values for the cells
    tempCells[i]->setTemperature(293.15 + i * 10.0);
    tempCells[i]->setVelocityU(i * 0.1);
    tempCells[i]->setVelocityV(i * -0.05);
  }

  // Test with all neighbors
  std::array<Cell *, 4> allNeighbors = {tempCells[0].get(), tempCells[1].get(),
                                        tempCells[2].get(), tempCells[3].get()};
  Cell::PropertyComputeFunction computeFunc =
      Cell::getPropertyComputation(propType);
  computeFunc(cell, &allNeighbors);
  results["all_neighbors"] = cell.getProperty(propType);

  // Test with each neighbor missing
  for (int i = 0; i < 4; i++) {
    std::array<Cell *, 4> neighbors = {nullptr, nullptr, nullptr, nullptr};
    for (int j = 0; j < 4; j++) {
      if (j != i)
        neighbors[j] = tempCells[j].get();
    }
    computeFunc(cell, &neighbors);
    results["missing_" + std::to_string(i)] = cell.getProperty(propType);
  }

  // Test with only one neighbor
  for (int i = 0; i < 4; i++) {
    std::array<Cell *, 4> neighbors = {nullptr, nullptr, nullptr, nullptr};
    neighbors[i] = tempCells[i].get();
    computeFunc(cell, &neighbors);
    results["only_" + std::to_string(i)] = cell.getProperty(propType);
  }

  // Test with no neighbors
  std::array<Cell *, 4> noNeighbors = {nullptr, nullptr, nullptr, nullptr};
  computeFunc(cell, &noNeighbors);
  results["no_neighbors"] = cell.getProperty(propType);

  return results;
}
