cmake_minimum_required(VERSION 3.14)
project(CFD_MeshKit VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default build type if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler warnings and options
if(MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

#[=======================================================================[.rst:
GoogleTest Configuration
-----------------------

This project uses GoogleTest for unit testing through CMake's FetchContent.
By default, both GoogleTest (GTest) and GoogleMock (GMock) are built.

GoogleTest Options
^^^^^^^^^^^^^^^^^

BUILD_GMOCK
  When ON, builds the GoogleMock library (default: ON).
  Set to OFF to build only GoogleTest without the mocking framework:
  
  .. code-block:: cmake
  
     set(BUILD_GMOCK OFF CACHE BOOL "Builds the googlemock subproject" FORCE)

gtest_force_shared_crt
  When ON, uses shared (DLL) run-time lib even when GoogleTest is built as static lib (default: OFF).
  This is necessary for Windows builds with MSVC to avoid runtime conflicts.
  
  .. code-block:: cmake
  
     set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

INSTALL_GTEST
  When ON, enables installation of GoogleTest (default: OFF).
  Usually not needed as GTest is only a test dependency.
  
  .. code-block:: cmake
  
     set(INSTALL_GTEST OFF CACHE BOOL "Enable installation of googletest" FORCE)

GTEST_CREATE_SHARED_LIBRARY
  When ON, builds GoogleTest as a shared library (default: OFF).
  
  .. code-block:: cmake
  
     set(GTEST_CREATE_SHARED_LIBRARY ON CACHE BOOL "" FORCE)

Test Organization
^^^^^^^^^^^^^^^^

Our tests are organized by functional area in the tests/CMakeLists.txt file.
To run specific test groups:

* All tests: ``ctest``
* Material tests only: ``ctest -R Material``

External References
^^^^^^^^^^^^^^^^^^

* GoogleTest GitHub: https://github.com/google/googletest
* GoogleTest Documentation: https://google.github.io/googletest/
* GoogleMock Documentation: https://google.github.io/googletest/gmock_for_dummies.html
* CMake FetchContent: https://cmake.org/cmake/help/latest/module/FetchContent.html
* CMake GoogleTest module: https://cmake.org/cmake/help/latest/module/GoogleTest.html
#]=======================================================================]

# Include directories
include_directories(
  ${PROJECT_SOURCE_DIR}/include
)

# First, build the utils library which has no dependencies
#add_subdirectory(utils)

# Build the MeshKit library
add_subdirectory(src)

# Testing with Google Test
option(BUILD_TESTS "Build the tests" ON)
if(BUILD_TESTS)
  # Enable testing
  enable_testing()
  
  # Download and build GoogleTest
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0  # More recent version with updated CMake requirements
  )
  # For Windows: Prevent overriding the parent project's compiler/linker settings
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  
  # Build tests
  add_subdirectory(tests)
endif()

# In the top-level CMakeLists.txt, add this after the Google Test section:

# Benchmarking with Google Benchmark
option(BUILD_BENCHMARKS "Build benchmarks" OFF)
if(BUILD_BENCHMARKS)
  # Download and build Google Benchmark
  include(FetchContent)
  FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.8.3  # Use a recent stable version
  )
  # Disable benchmark tests and install to keep it simpler
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark testing" FORCE)
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "Disable benchmark install" FORCE)
  FetchContent_MakeAvailable(benchmark)
  
  # Build benchmarks
  add_subdirectory(benchmarks)
endif()

# Installation rules
install(TARGETS meshkit
  EXPORT meshkit-targets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

# Get all header files
file(GLOB MESHKIT_HEADERS "${PROJECT_SOURCE_DIR}/include/*.h")
install(FILES ${MESHKIT_HEADERS} DESTINATION include)

install(EXPORT meshkit-targets
  FILE meshkit-targets.cmake
  NAMESPACE meshkit::
  DESTINATION lib/cmake/meshkit
)

# Generate and install package configuration files
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/meshkit-config-version.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

# Update meshkit-config.cmake.in to reference the correct utilities target files
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/meshkit-config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/meshkit-config.cmake"
  @ONLY
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/meshkit-config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/meshkit-config-version.cmake"
  DESTINATION lib/cmake/meshkit
)

# Add examples if they exist
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples")
  option(BUILD_EXAMPLES "Build example applications" OFF)
  if(BUILD_EXAMPLES)
    add_subdirectory(examples)
  endif()
endif()
