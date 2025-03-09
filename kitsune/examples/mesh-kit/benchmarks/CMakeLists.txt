# Benchmarks directory

# Group benchmark files by functional area
set(MATERIAL_BENCHMARKS
	#material_basics_benchmarks.cpp
	# material_cache_benchmarks.cpp
)

set(MESH_BENCHMARKS
    region_bitset_benchmarks.cpp
    # Add mesh-related benchmark files as they are created
    # mesh_benchmarks.cpp
)

set(FLOW_BENCHMARKS
    # Add flow-related benchmark files as they are created
    # flow_benchmarks.cpp
)

# Collect all benchmark source files
set(BENCHMARK_SOURCES
    ${MATERIAL_BENCHMARKS}
    ${MESH_BENCHMARKS}
    ${FLOW_BENCHMARKS}
)

# Create the benchmark executable
add_executable(meshkit_benchmarks ${BENCHMARK_SOURCES})

# Link with the main library and benchmark framework
target_link_libraries(meshkit_benchmarks PRIVATE
    meshkit
    benchmark::benchmark
    pthread  # Required for benchmarks on many systems
)

# Add source files to IDE projects with virtual folders
source_group("Material Benchmarks" FILES ${MATERIAL_BENCHMARKS})
source_group("Mesh Benchmarks" FILES ${MESH_BENCHMARKS})
source_group("Flow Benchmarks" FILES ${FLOW_BENCHMARKS})
