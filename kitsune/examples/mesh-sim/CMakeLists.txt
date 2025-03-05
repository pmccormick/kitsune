cmake_minimum_required(VERSION 3.10)

# Project name
project(CFDTests VERSION 0.01)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# GCC/Clang compiler flags
add_compile_options(-Wall -Wextra -Wpedantic)

# Enable testing
enable_testing()

# Add source files for the library
set(LIB_SOURCES
    src/BoundaryFactory.cpp
    src/Cell.cpp    
    src/DirichletBoundary.cpp
    src/InflowBoundary.cpp
    src/NeumannBoundary.cpp
    src/NoSlipBoundary.cpp
    src/PeriodicBoundary.cpp
    src/SlipBoundary.cpp    
    src/CellExtensions.cpp    
    src/Material.cpp
    # Add other source files as needed
)

# Add header files for the library
set(LIB_HEADERS
    include/BoundaryClass.h	
    include/BoundaryFactory.h
    include/Cell.h
    include/DirichletBoundary.h
    include/InflowBoundary.h
    include/Material.h
    include/NeumannBoundary.h
    include/NoSlipBoundary.h
    include/PeriodicBoundary.h
    include/SlipBoundary.h
    include/Units.h
    # Add other header files as needed
)

# Create library
add_library(CFDLib ${LIB_SOURCES})

# Set include directories for the library
target_include_directories(CFDLib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Add test executables

# Cell tests
add_executable(cell_tests tests/cell_test_suite.cpp)
target_link_libraries(cell_tests PRIVATE CFDLib)
add_test(NAME CellTests COMMAND cell_tests)

# Cell units tests
add_executable(cell_units_tests tests/tests_for_cell_units.cpp)
target_link_libraries(cell_units_tests PRIVATE CFDLib)
add_test(NAME CellUnitsTests COMMAND cell_units_tests)

# Material tests
add_executable(material_tests tests/material_test_suite.cpp)
target_link_libraries(material_tests PRIVATE CFDLib)
add_test(NAME MaterialTests COMMAND material_tests)

# Material mixing tests
add_executable(material_mixing_tests tests/material_mixing_tests.cpp)
target_link_libraries(material_mixing_tests PRIVATE CFDLib)
add_test(NAME MaterialMixingTests COMMAND material_mixing_tests)

# Bondary condition tests
add_executable(inflow_boundary_tests tests/inflow_boundary_tests.cpp)
target_link_libraries(inflow_boundary_tests PRIVATE CFDLib)
add_test(NAME InflowBoundaryTests COMMAND inflow_boundary_tests)

add_executable(slip_boundary_tests tests/slip_boundary_tests.cpp)
target_link_libraries(slip_boundary_tests PRIVATE CFDLib)
add_test(NAME SlipBoundaryTests COMMAND slip_boundary_tests)

add_executable(neumann_boundary_tests tests/neumann_boundary_tests.cpp)
target_link_libraries(neumann_boundary_tests PRIVATE CFDLib)
add_test(NAME NeumannBoundaryTests COMMAND neumann_boundary_tests)

add_executable(periodic_boundary_tests tests/periodic_boundary_tests.cpp)
target_link_libraries(periodic_boundary_tests PRIVATE CFDLib)
add_test(NAME PeriodicBoundaryTests COMMAND periodic_boundary_tests)

# Optional performance tests (build only when specifically requested)
option(BUILD_PERF_TESTS "Build performance tests" OFF)
if(BUILD_PERF_TESTS)
    add_executable(cell_perf_tests tests/cell_performance_analysis.cpp)
    target_link_libraries(cell_perf_tests PRIVATE CFDLib)
endif()

# Install targets
install(TARGETS CFDLib
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES ${LIB_HEADERS} DESTINATION include)

# Print configuration information
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")

