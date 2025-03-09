# MeshKit tests directory

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

# MeshKit tests directory

# Group test files by functional area
set(MATERIAL_TESTS
    cell_basic_tests.cpp
    mesh_base_tests.cpp
    field_basic_tests.cpp
    field_iterator_tests.cpp
    mesh_accessor_iterator_tests.cpp
    material_tests.cpp
    material_copy_move_tests.cpp
    material_multi_mixing_tests.cpp
    material_simple_mixing_tests.cpp
    region_debug_tests.cpp
    region_simple_tests.cpp
    scientific_mequiv_basic_tests.cpp
    scientific_mequiv_advanced_industry_tests.cpp
    scientific_mequiv_resolution_tests.cpp
    scientific_mequiv_hash_distribution_tests.cpp
    scientific_mequiv_robustness_tests.cpp
    scientific_mequiv_stability_tests.cpp
    scientific_mequiv_enhanced_numerical_tests.cpp
    scientific_mequiv_simulation_stress_test.cpp
)

set(MESH_TESTS
    # Add mesh-related test files as they are created
    # mesh_tests.cpp
)

set(FLOW_TESTS
    # Add flow-related test files as they are created
    # flow_initializer_tests.cpp
    # obstacle_generator_tests.cpp
)

# Collect all test source files
set(TEST_SOURCES
    ${MATERIAL_TESTS}
    ${MESH_TESTS}
    ${FLOW_TESTS}
)

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/MaterialCache
)
# Create the main test executable
add_executable(meshkit_tests ${TEST_SOURCES})

# Link with the main library and testing framework
target_link_libraries(meshkit_tests PRIVATE
    meshkit
    gtest
    gtest_main
)

# Add source files to IDE projects with virtual folders
source_group("Material Tests" FILES ${MATERIAL_TESTS})
source_group("Mesh Tests" FILES ${MESH_TESTS})
source_group("Flow Tests" FILES ${FLOW_TESTS})

# Register tests with CTest
include(GoogleTest)
gtest_discover_tests(meshkit_tests)

# Add individual test targets for specific test groups if needed
# This allows running only specific test groups using "ctest -R Material"
add_test(NAME MaterialTests COMMAND meshkit_tests --gtest_filter="MaterialTest*")
# add_test(NAME MeshTests COMMAND meshkit_tests --gtest_filter="MeshTest*")
# add_test(NAME FlowTests COMMAND meshkit_tests --gtest_filter="FlowTest*")
