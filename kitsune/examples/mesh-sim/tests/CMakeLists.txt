cmake_minimum_required(VERSION 3.10)
project(CFDSimulationTests)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Google Test
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Include your project source directory
include_directories(${PROJECT_SOURCE_DIR}/src)

# Add test executables
add_executable(cell_tests CellTest.cpp Cell.cpp Material.cpp)
target_link_libraries(cell_tests ${GTEST_LIBRARIES} pthread)

add_executable(material_tests MaterialTest.cpp Material.cpp)
target_link_libraries(material_tests ${GTEST_LIBRARIES} pthread)

# Add a custom target to run all tests
add_custom_target(run_tests
    COMMAND cell_tests
    COMMAND material_tests
    DEPENDS cell_tests material_tests
)

