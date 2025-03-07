/**
 * @file cell_test_fixture.h
 * @brief Test fixture for Cell tests with field-based storage
 *
 * This fixture creates a small mesh with initialized fields
 * to test the Cell class behavior in isolation.
 */

#ifndef CELL_TEST_FIXTURE_H
#define CELL_TEST_FIXTURE_H

#include "Cell.h"
#include "Material.h"
#include "Mesh.h"

#include <array>
#include <memory>
#include <string>

/**
 * @class CellTestFixture
 * @brief Provides a consistent test environment for Cell tests
 *
 * The fixture creates a small mesh with predictable values in all fields
 * and provides helper methods to access cells in standard test positions.
 */
class CellTestFixture {
public:
  /**
   * @brief Constructor - initializes a 3x3 mesh with predictable field values
   */
  CellTestFixture()
      : mesh(3, 3, 1.0, 1.0) // 3x3 mesh with 1.0x1.0 dimensions
  {
    // Initialize a default material
    defaultMaterial = std::make_shared<Material>(Material::MaterialType::FLUID,
                                                 "DefaultTestMaterial");
    defaultMaterial->setProperty(Material::MaterialProperty::DENSITY, 1.0);
    defaultMaterial->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY,
                                 0.01);
    defaultMaterial->setProperty(
        Material::MaterialProperty::THERMAL_CONDUCTIVITY, 0.5);
    defaultMaterial->setProperty(Material::MaterialProperty::SPECIFIC_HEAT,
                                 1000.0);

    // Set values in the mesh
    initializeMesh();
  }

  /**
   * @brief Get the central fluid cell (1,1) for most tests
   * @return Reference to the central cell
   */
  Cell &getCentralCell() { return mesh.getCell(1, 1); }

  /**
   * @brief Get a boundary cell at (0,1) for boundary tests
   * @return Reference to the west boundary cell
   */
  Cell &getWestBoundaryCell() { return mesh.getCell(0, 1); }

  /**
   * @brief Get a boundary cell at (2,1) for boundary tests
   * @return Reference to the east boundary cell
   */
  Cell &getEastBoundaryCell() { return mesh.getCell(2, 1); }

  /**
   * @brief Get a boundary cell at (1,0) for boundary tests
   * @return Reference to the south boundary cell
   */
  Cell &getSouthBoundaryCell() { return mesh.getCell(1, 0); }

  /**
   * @brief Get a boundary cell at (1,2) for boundary tests
   * @return Reference to the north boundary cell
   */
  Cell &getNorthBoundaryCell() { return mesh.getCell(1, 2); }

  /**
   * @brief Get an array of the four cells surrounding the central cell
   * @return Array of pointers to surrounding cells in N,E,S,W order
   */
  std::array<Cell *, 4> getSurroundingCells() {
    std::array<Cell *, 4> cells = {
        &mesh.getCell(1, 2), // North
        &mesh.getCell(2, 1), // East
        &mesh.getCell(1, 0), // South
        &mesh.getCell(0, 1)  // West
    };
    return cells;
  }

  /**
   * @brief Access to the mesh for direct field manipulation
   * @return Reference to the test mesh
   */
  Mesh &getMesh() { return mesh; }

  /**
   * @brief Get the default material used in the test
   * @return Shared pointer to the default material
   */
  std::shared_ptr<Material> getDefaultMaterial() { return defaultMaterial; }

private:
  /**
   * @brief Initialize the mesh with predictable test values
   */
  void initializeMesh() {
    // Register the default material with all cells
    for (size_t j = 0; j < mesh.getNy(); ++j) {
      for (size_t i = 0; i < mesh.getNx(); ++i) {
        mesh.setCellMaterial(i, j, defaultMaterial);
      }
    }

    // Initialize field values
    for (size_t j = 0; j < mesh.getNy(); ++j) {
      for (size_t i = 0; i < mesh.getNx(); ++i) {
        // Temperature (K) - base 300K + position-based offset
        mesh.getTemperatureField()(i, j) = 300.0 + (i + j);

        // Pressure (Pa) - base 101325 Pa + position-based offset
        mesh.getPressureField()(i, j) = 101325.0 + 100.0 * (i + j);

        // Density (kg/m³) - base 1.0 + small position offset
        mesh.getDensityField()(i, j) = 1.0 + 0.01 * (i + j);

        // Velocity (m/s)
        mesh.getVelocityUField()(i, j) = 0.1 * i;
        mesh.getVelocityVField()(i, j) = 0.1 * j;

        // Initialize properties
        for (size_t p = 0; p < static_cast<size_t>(Cell::PropertyType::COUNT);
             ++p) {
          mesh.getPropertiesField()(i, j, p) = p * 10.0 + i + j;
        }

        // Initialize vertex velocities
        for (size_t v = 0; v < 4; ++v) {
          mesh.getVertexVelocityXField()(i, j, v) = 0.05 * (i + v);
          mesh.getVertexVelocityYField()(i, j, v) = 0.05 * (j + v);
        }
      }
    }

    // Set types for boundary cells (already done by the mesh constructor,
    // but we make it explicit here for test clarity)
    for (size_t j = 0; j < mesh.getNy(); ++j) {
      for (size_t i = 0; i < mesh.getNx(); ++i) {
        if (i == 0 || i == mesh.getNx() - 1 || j == 0 ||
            j == mesh.getNy() - 1) {
          mesh.setCellAsBoundary(i, j);
        } else {
          mesh.setCellAsFluid(i, j, defaultMaterial);
        }
      }
    }
  }

  // The test mesh
  Mesh mesh;

  // Default material for all cells
  std::shared_ptr<Material> defaultMaterial;
};

#endif // CELL_TEST_FIXTURE_H
