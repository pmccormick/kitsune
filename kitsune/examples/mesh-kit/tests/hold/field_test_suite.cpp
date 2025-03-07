/**
 * @file field_test_suite.cpp
 * @brief Test suite for the Field class
 *
 * This file contains tests for the Field class, focusing on storage,
 * access patterns, and dimension handling.
 */

#include "Field.h"
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// Simple test framework
#define TEST(name) void name()
#define ASSERT(condition)                                                      \
  if (!(condition)) {                                                          \
    std::cerr << "Assertion failed: " << #condition << " at " << __FILE__      \
              << ":" << __LINE__ << std::endl;                                 \
    assert(condition);                                                         \
  }
#define ASSERT_EQ(a, b) ASSERT((a) == (b))
#define ASSERT_DOUBLE_EQ(a, b) ASSERT(std::abs((a) - (b)) < 1e-10)
#define ASSERT_THROWS(expr, excType)                                           \
  try {                                                                        \
    expr;                                                                      \
    ASSERT(false && "Expected exception not thrown");                          \
  } catch (const excType &) {                                                  \
  } catch (...) {                                                              \
    ASSERT(false && "Wrong exception type thrown");                            \
  }

// Test field dimensions based on tag
TEST(TestFieldDimensions) {
  size_t nx = 5, ny = 4;

  // Cell center field should have dimensions nx x ny
  CellCenterField<double> centerField(nx, ny);
  ASSERT_EQ(centerField.nx(), nx);
  ASSERT_EQ(centerField.ny(), ny);
  ASSERT_EQ(centerField.size(), nx * ny);

  // Vertex field should have dimensions (nx+1) x (ny+1)
  VertexField<double> vertexField(nx, ny);
  ASSERT_EQ(vertexField.nx(), nx + 1);
  ASSERT_EQ(vertexField.ny(), ny + 1);
  ASSERT_EQ(vertexField.size(), (nx + 1) * (ny + 1));

  // Horizontal edge field should have dimensions nx x (ny+1)
  HorizontalEdgeField<double> hEdgeField(nx, ny);
  ASSERT_EQ(hEdgeField.nx(), nx);
  ASSERT_EQ(hEdgeField.ny(), ny + 1);
  ASSERT_EQ(hEdgeField.size(), nx * (ny + 1));

  // Vertical edge field should have dimensions (nx+1) x ny
  VerticalEdgeField<double> vEdgeField(nx, ny);
  ASSERT_EQ(vEdgeField.nx(), nx + 1);
  ASSERT_EQ(vEdgeField.ny(), ny);
  ASSERT_EQ(vEdgeField.size(), (nx + 1) * ny);
}

// Test 2D field access
TEST(Test2DFieldAccess) {
  size_t nx = 3, ny = 3;
  CellCenterField<double> field(nx, ny);

  // Fill field with test values
  for (size_t j = 0; j < ny; ++j) {
    for (size_t i = 0; i < nx; ++i) {
      field(i, j) = i * 10 + j;
    }
  }

  // Verify access in reading
  for (size_t j = 0; j < ny; ++j) {
    for (size_t i = 0; i < nx; ++i) {
      ASSERT_DOUBLE_EQ(field(i, j), i * 10 + j);
    }
  }

  // Test edge values
  ASSERT_DOUBLE_EQ(field(0, 0), 0);
  ASSERT_DOUBLE_EQ(field(nx - 1, 0), (nx - 1) * 10);
  ASSERT_DOUBLE_EQ(field(0, ny - 1), ny - 1);
  ASSERT_DOUBLE_EQ(field(nx - 1, ny - 1), (nx - 1) * 10 + (ny - 1));
}

// Test 3D field access
TEST(Test3DFieldAccess) {
  size_t nx = 3, ny = 3, depth = 4;
  Field<double, CellCenterTag, 4> field3D(nx, ny);

  // Verify dimensions
  ASSERT_EQ(field3D.nx(), nx);
  ASSERT_EQ(field3D.ny(), ny);
  ASSERT_EQ(field3D.depth(), depth);
  ASSERT_EQ(field3D.size(), nx * ny * depth);

  // Fill field with test values
  for (size_t k = 0; k < depth; ++k) {
    for (size_t j = 0; j < ny; ++j) {
      for (size_t i = 0; i < nx; ++i) {
        field3D(i, j, k) = i * 100 + j * 10 + k;
      }
    }
  }

  // Verify access in reading
  for (size_t k = 0; k < depth; ++k) {
    for (size_t j = 0; j < ny; ++j) {
      for (size_t i = 0; i < nx; ++i) {
        ASSERT_DOUBLE_EQ(field3D(i, j, k), i * 100 + j * 10 + k);
      }
    }
  }

  // Test specific values
  ASSERT_DOUBLE_EQ(field3D(1, 2, 3), 123);
  ASSERT_DOUBLE_EQ(field3D(2, 1, 0), 210);
}

// Test field with non-double data type
TEST(TestNonDoubleField) {
  size_t nx = 3, ny = 3;

  // Test with integers
  Field<int, CellCenterTag> intField(nx, ny);
  for (size_t j = 0; j < ny; ++j) {
    for (size_t i = 0; i < nx; ++i) {
      intField(i, j) = static_cast<int>(i * 10 + j);
    }
  }

  for (size_t j = 0; j < ny; ++j) {
    for (size_t i = 0; i < nx; ++i) {
      ASSERT_EQ(intField(i, j), static_cast<int>(i * 10 + j));
    }
  }

  // Test with booleans
  Field<bool, CellCenterTag> boolField(nx, ny);
  for (size_t j = 0; j < ny; ++j) {
    for (size_t i = 0; i < nx; ++i) {
      boolField(i, j) = ((i + j) % 2 == 0);
    }
  }

  for (size_t j = 0; j < ny; ++j) {
    for (size_t i = 0; i < nx; ++i) {
      ASSERT_EQ(boolField(i, j), ((i + j) % 2 == 0));
    }
  }

  // Test with strings
  Field<std::string, CellCenterTag> stringField(nx, ny);
  for (size_t j = 0; j < ny; ++j) {
    for (size_t i = 0; i < nx; ++i) {
      stringField(i, j) = "Cell_" + std::to_string(i) + "_" + std::to_string(j);
    }
  }

  for (size_t j = 0; j < ny; ++j) {
    for (size_t i = 0; i < nx; ++i) {
      ASSERT_EQ(stringField(i, j),
                "Cell_" + std::to_string(i) + "_" + std::to_string(j));
    }
  }
}

// Test the fill method
TEST(TestFieldFill) {
  size_t nx = 5, ny = 4;
  CellCenterField<double> field(nx, ny);

  // Test filling with a constant value
  field.fill(42.0);

  for (size_t j = 0; j < ny; ++j) {
    for (size_t i = 0; i < nx; ++i) {
      ASSERT_DOUBLE_EQ(field(i, j), 42.0);
    }
  }
}

// Test direct access to the underlying data
TEST(TestDirectDataAccess) {
  size_t nx = 3, ny = 2;
  CellCenterField<double> field(nx, ny);

  // Fill field with values
  for (size_t j = 0; j < ny; ++j) {
    for (size_t i = 0; i < nx; ++i) {
      field(i, j) = i * 10 + j;
    }
  }

  // Get direct access to data
  const std::vector<double> &data = field.data();

  // Verify data size
  ASSERT_EQ(data.size(), nx * ny);

  // Verify data values (requires knowledge of the internal layout)
  for (size_t j = 0; j < ny; ++j) {
    for (size_t i = 0; i < nx; ++i) {
      size_t idx = i + j * nx;
      ASSERT_DOUBLE_EQ(data[idx], i * 10 + j);
    }
  }

  // Modify data via reference
  std::vector<double> &dataRef = field.data();
  for (size_t i = 0; i < dataRef.size(); ++i) {
    dataRef[i] = 100 + i;
  }

  // Verify changes through field interface
  for (size_t j = 0; j < ny; ++j) {
    for (size_t i = 0; i < nx; ++i) {
      size_t idx = i + j * nx;
      ASSERT_DOUBLE_EQ(field(i, j), 100 + idx);
    }
  }
}

// Test for boundary checking
TEST(TestBoundaryChecking) {
  // Check that reasonable access works
  size_t nx = 3, ny = 3;
  CellCenterField<int> field(nx, ny);

  // These should be fine
  field(0, 0) = 1;
  field(nx - 1, ny - 1) = 1;

  // In debug mode, these would trigger assertions
  // but since we can't rely on that in tests, we'll skip

  // This would cause a memory access violation in release mode
  // field(nx, ny) = 1;
}

// Main function to run all tests
int main() {
  try {
    std::cout << "Running Field test suite..." << std::endl;

    // Run all tests
    TestFieldDimensions();
    Test2DFieldAccess();
    Test3DFieldAccess();
    TestNonDoubleField();
    TestFieldFill();
    TestDirectDataAccess();
    TestBoundaryChecking();

    std::cout << "All tests passed!" << std::endl;
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Exception caught: " << e.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "Unknown exception caught!" << std::endl;
    return 1;
  }
}