# Build and Test Instructions for CFD Cell and Material Tests

This document provides instructions on how to build and run the test suite for the Cell and Material classes using CMake.

## Prerequisites

- CMake 3.10 or higher
- C++17 compatible compiler (GCC, Clang, MSVC)
- Git (optional, for version control)

## Project Structure

The project follows a standard structure:

```
CFDTests/
├── CMakeLists.txt          # Main CMake configuration
├── include/                # Header files
│   ├── Cell.h
│   ├── Material.h
│   └── Units.h
├── src/                    # Implementation files
│   ├── Cell.cpp
│   └── Material.cpp
└── tests/                  # Test files
    ├── cell_test_suite.cpp
    ├── unit_tests_for_cell_units.cpp
    ├── material_test_suite.cpp
    └── cell_performance_analysis.cpp
```

## Building the Project

### Basic Build

```bash
# Create a build directory
mkdir build
cd build

# Configure with CMake
cmake ..

# Build the project
cmake --build .    # or simply 'make' on Unix-like systems
```

### Build with Performance Tests

Performance tests are disabled by default to speed up normal testing. To build with performance tests:

```bash
mkdir build
cd build
cmake .. -DBUILD_PERF_TESTS=ON
cmake --build .
```

### Build Types

You can specify the build type (Debug/Release):

```bash
# For debug build
cmake .. -DCMAKE_BUILD_TYPE=Debug

# For release build
cmake .. -DCMAKE_BUILD_TYPE=Release
```

## Running Tests

### Running All Tests

After building, you can run all tests using CTest:

```bash
cd build
ctest
```

### Running Individual Tests

You can also run specific test executables directly:

```bash
# Run Cell tests
./cell_tests

# Run Cell Units tests
./cell_units_tests

# Run Material tests
./material_tests

# Run performance tests (if built)
./cell_perf_tests
```

## Troubleshooting

### Common Issues

1. **CMake Error: Could not find a package configuration file**
   - Ensure all required dependencies are installed

2. **Compilation Errors**
   - Check that your compiler supports C++17
   - Verify that the header files are in the correct location

3. **Test Failures**
   - Check output for specific test failures
   - Verify that tolerance values in tests are appropriate for your implementation

### Debug Test Failures

If tests are failing, you can add more verbose output:

```bash
# Run CTest with verbose output
ctest -V

# Or run the specific test executable with more debug output
./cell_tests
```

## Additional Notes

- The tests include tolerance values for floating-point comparisons. If your implementation uses slightly different constants for unit conversions, you may need to adjust these tolerance values.
- If you encounter any "symbol not found" errors, make sure your library implementation correctly exports all required functions.

