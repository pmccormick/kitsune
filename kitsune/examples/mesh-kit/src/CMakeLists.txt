# MeshKit source directory

# Material system files
set(MATERIAL_SOURCES
    Material.cpp
    MaterialCache/HybridEquivalence.cpp
    MaterialCache/MaterialCache.cpp
    MaterialCache/MaterialEquivalence.cpp
    MaterialCache/ScientificMaterialEquivalence.cpp
    MaterialCache/StandardEquivalence.cpp
    MaterialCache/ULPEquivalence.cpp 
)

# Mesh system files
set(MESH_SOURCES
    Cell.cpp 
    MeshBase.cpp
    Mesh.cpp
    Region.cpp
    # Add mesh-related files as they are added to the project
    # Mesh.cpp
    # MeshElement.cpp
)

# Flow utilities and generation
set(FLOW_SOURCES
    #FlowInitializer.cpp
    #ObstacleGenerator.cpp
)

# Collect all source files
set(MESHKIT_SOURCES
    ${MATERIAL_SOURCES}
    ${MESH_SOURCES}
    ${FLOW_SOURCES}
)

# Create the MeshKit library
add_library(meshkit STATIC ${MESHKIT_SOURCES})

# Set include directories
target_include_directories(meshkit PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/MaterialCache>
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
