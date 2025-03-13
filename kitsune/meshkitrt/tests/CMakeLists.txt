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

# Mock support files (reusable across test groups)
set(MOCK_SUPPORT
   # 
)
 

# Group test files by functional area

set(BITARRAY_TESTS
   bit-array-bitwise-test.cpp
   bit-array-construction-test.cpp
   bit-array-counting-test.cpp
   bit-array-edge-case-test.cpp
   bit-array-manipulation-test.cpp
   bit-array-performance-test.cpp
   bit-array-resize-test.cpp
   bit-array-traversal-test.cpp
)

set(CELLS_TESTS
   cell-basic-tests.cpp
   cell-edge-case-tests.cpp
   cell-mesh-integration-tests.cpp
   cell-performance-tests.cpp
   cell-boundary-tests.cpp
   cell-index-tests.cpp
   cell-navigation-tests.cpp
)


set(FIELD_TESTS 
)

set(MESH_TESTS 
   mesh-basic-tests.cpp
   mesh-cell-factory-tests.cpp
   mesh-index-tests.cpp
   mesh-integration-tests.cpp
   mesh-test-main.cpp
)


set(MATERIAL_TESTS
	#   material_tests.cpp
	#   material_copy_move_tests.cpp
	#   material_multi_mixing_tests.cpp
	#   material_simple_mixing_tests.cpp
	#   scientific_mequiv_basic_tests.cpp
	#   scientific_mequiv_advanced_industry_tests.cpp
	#   scientific_mequiv_resolution_tests.cpp
	#   scientific_mequiv_hash_distribution_tests.cpp
	#   scientific_mequiv_robustness_tests.cpp
	#   scientific_mequiv_stability_tests.cpp
	#   scientific_mequiv_enhanced_numerical_tests.cpp
	#   scientific_mequiv_simulation_stress_test.cpp
)

set(REGION_TESTS
	# region_basic_tests.cpp
	# region_advanced_tests.cpp
)

# Collect all test source files
set(TEST_SOURCES
   ${BITARRAY_TESTS}
   ${CELL_TESTS}
   ${FIELD_TESTS}
   ${MATERIAL_TESTS}
   ${REGION_TESTS}
)

# Create a STATIC library with the mock objects for potential reuse
#add_library(test_mocks STATIC ${MOCK_SUPPORT})
#target_include_directories(test_mocks PUBLIC 
#    ${CMAKE_SOURCE_DIR}/include
#    ${CMAKE_SOURCE_DIR}/include/Material
#    ${CMAKE_SOURCE_DIR}/include/Mesh
#    ${CMAKE_SOURCE_DIR}/include/Region
#    ${CMAKE_CURRENT_SOURCE_DIR}  # Include current directory for mock headers
#)
#target_link_libraries(test_mocks PUBLIC meshkit)

# Set includes for all test files
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/../include
    ${CMAKE_SOURCE_DIR}/../include/Material
    ${CMAKE_SOURCE_DIR}/../include/Mesh
    ${CMAKE_SOURCE_DIR}/../include/Region
    ${CMAKE_CURRENT_SOURCE_DIR}  # Include current directory for mock headers
)

# Create the main test executable
add_executable(meshkit_tests ${TEST_SOURCES})

# Link with the main library, mock library, and testing framework
target_link_libraries(meshkit_tests PRIVATE
    meshkit
    gtest
    gtest_main
)

# Add source files to IDE projects with virtual folders
source_group("Test Support" FILES ${MOCK_SUPPORT})
source_group("BitArray Tests" FILES ${BITARRAY_TESTS})
source_group("Cell Tests" FILES ${CELL_TESTS})
#source_group("Field Tests" FILES ${MESH_TESTS})
source_group("Region Tests" FILES ${REGION_TESTS})

# Register tests with CTest
include(GoogleTest)
gtest_discover_tests(meshkit_tests)

# Add individual test targets for specific test groups
add_test(NAME MaterialTests COMMAND meshkit_tests --gtest_filter="MaterialTest*")
# add_test(NAME MeshTests COMMAND meshkit_tests --gtest_filter="MeshTest*")
# add_test(NAME FlowTests COMMAND meshkit_tests --gtest_filter="FlowTest*")
add_test(NAME RegionTests COMMAND meshkit_tests --gtest_filter="RegionTest*")

# Optional: For standalone test executables (e.g., for debugging specific test groups)
# This allows building just a specific test suite for more focused testing/debugging
option(BUILD_STANDALONE_TESTS "Build standalone test executables for each test group" OFF)

if(BUILD_STANDALONE_TESTS)
    # Standalone Region Tests executable
    add_executable(region_tests ${REGION_TESTS})
    target_link_libraries(region_tests PRIVATE
        meshkit
        gtest
        gtest_main
    )
endif()

