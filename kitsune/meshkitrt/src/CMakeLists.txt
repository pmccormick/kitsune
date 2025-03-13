# MeshKit source directory

# Material system files
set(MATERIAL_SOURCES
    Material/HybridEquivalence.cpp
    Material/MaterialCache.cpp
    Material/MaterialEquivalence.cpp
    Material/ScientificMaterialEquivalence.cpp
    Material/StandardEquivalence.cpp
    Material/ULPEquivalence.cpp 
)

# Mesh system files
set(MESH_SOURCES
   Mesh/CellImpl.cpp
)

# Region system files
set(REGION_SOURCES
   Region/Region.cpp
   Region/RegionDefinition.cpp
   Region/RegionOperations.cpp
   Region/RegionOptimizations.cpp
   Region/RegionStorage.cpp
   Region/RegionUtils.cpp
)

# Collect all source files
set(MESHKIT_SOURCES
    ${MATERIAL_SOURCES}
    ${MESH_SOURCES}
    ${REGION_SOURCES}
)

# Create the MeshKit library
add_library(meshkit STATIC ${MESHKIT_SOURCES})

# Set include directories
target_include_directories(meshkit PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/Material>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/Mesh>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/Region>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/Units>
  $<INSTALL_INTERFACE:include>
)

# Link with the utilities library
#target_link_libraries(meshkit PUBLIC meshkit_utils)

# Add source files to IDE projects with virtual folders
source_group("Material" FILES ${MATERIAL_SOURCES})
source_group("Mesh" FILES ${MESH_SOURCES})
source_group("Flow" FILES ${FLOW_SOURCES})

# Set properties
set_target_properties(meshkit PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
  POSITION_INDEPENDENT_CODE ON
)

# Output the source files for debugging
message(STATUS "Building MeshKit with the following sources:")
foreach(src ${MESHKIT_SOURCES})
  message(STATUS "  - ${src}")
endforeach()
