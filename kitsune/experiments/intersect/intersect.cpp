//
// Copyright(c) 2020 Triad National Security, LLC
// All rights reserved.
//
// This file is part of the kitsune / llvm project.    It is released under
// the LLVM license.
//
// Simple example of mesh intersection. Hopefully represents relevant memory
// access patterns.
//
#include <cstdio>
#include <cstring>
#include <stdlib.h>
#include <string>
#include <kitsune.h>
#include "kitsune/kitrt/llvm-gpu.h"
#include "kitsune/kitrt/kitrt-cuda.h"
#include <nvToolsExt.h>

#include "common.h"
#include "serial_algorithms.h"

const bool CHECK = true;

using namespace std;

int main(int argc, char *argv[]) {

  // for profiling, do all the cuda overhead now
  nvtxRangePush("Kitsune Initialization");
  int *dum = (int *)__kitrt_cuMemAllocManaged(sizeof(int) * 16);
  //   __kitrt_cuMemFree(dum); // doesn't work ATM
  nvtxRangePop();

  ///////////////////////////////
  // simulation parameters
  ///////////////////////////////

  int nx = 3, ny = 3;
  const double x_max = 1., y_max = 1., shift_x = .0, shift_y = .0;
  bool shuffle = false, use_UVM = false;
  PrefetchKinds PFKind = NONE;

  parse_args(argc, argv, &nx, &ny, &shuffle, &PFKind);

  // derived parameters
  const int n_cells = nx * ny, n_nodes = (nx + 1) * (ny + 1);

  printf("x_max=%f, y_max=%f\nshift_x=%f, shift_y=%f\n", x_max, y_max, shift_x,
         shift_y);

  ///////////////////////////////
  // create the mesh
  ///////////////////////////////

  nvtxRangePush("Mesh Creation");

  // declare source, target meshes, and intersection candidates
  double *source_coordinates, *target_coordinates;
  size_t *source_cell_nodes, *source_node_offsets;
  size_t *target_cell_nodes, *target_node_offsets;
  size_t *candidates, *candidate_offsets;

  create_meshes(HOST, nx, ny, x_max, y_max, shift_x, shift_y,
                source_coordinates, source_cell_nodes, source_node_offsets,
                target_coordinates, target_cell_nodes, target_node_offsets,
                candidates, candidate_offsets, shuffle);
  nvtxRangePop();

  ///////////////////////////////
  // do serial work
  ///////////////////////////////

  nvtxRangePush("Centroid: Serial Computation");
  double *source_centroids =
      allocate<double>(HOST, 2 * n_cells, "source centroids");
  nvtxMark("starting centroid computation");
  for (size_t i = 0; i < n_cells; ++i)
    serial::centroid(i, source_node_offsets, source_cell_nodes,
                     source_coordinates, source_centroids);
  if constexpr (LOG_LEVEL > 0) {
    printf("\nComputed centroids\n");
    for (size_t i = 0; i < n_cells; ++i)
      printf("cell %3zu centroid: %.4f, %.4f\n", i, source_centroids[2 * i],
             source_centroids[2 * i + 1]);
  }
  nvtxRangePop();

  nvtxRangePush("Plane distance: Serial Computation");
  nvtxMark("starting plane distance computation");

  // first pass to determine sizes
  // loop over target cells and then candidates
  // multiply the number of nodes in each and accumulate
  size_t n_plane_distances = 0;
  for (size_t i = 0; i < n_cells; ++i) {
    size_t n_target_nodes = target_node_offsets[i + 1] - target_node_offsets[i];
    LOG_LEVEL > 1 &&
        printf("target cell %lu has %lu nodes\n", i, n_target_nodes);
    // loop over neighboring source cells
    for (size_t cci = candidate_offsets[i]; cci < candidate_offsets[i + 1];
         ++cci) {
      size_t candidate = candidates[cci];
      size_t n_source_nodes =
          source_node_offsets[candidate + 1] - source_node_offsets[candidate];
      LOG_LEVEL > 1 &&
          printf("  source cell %lu has %lu nodes\n", cci, n_source_nodes);
      n_plane_distances += n_source_nodes * n_target_nodes;
    }
  }
  LOG_LEVEL > 0 && printf("there are %lu candidate distances to compute\n",
                          n_plane_distances);

  // allocate the distance array and reset the counter
  double *distances =
      allocate<double>(HOST, n_plane_distances, "plane distance");

  // DWS THIS WON'T WORK IN PARALLEL, NEED TO USE OFFSETS
  n_plane_distances = 0;

  // in the following loops the candidates depend on the target cell but
  // the creation of both source and target edges and triangles can be moved
  // around and it is unclear where the best place for them is

  // loop over target cells
  for (size_t i = 0; i < n_cells; ++i) {
    // LOG_LEVEL > 1 && printf("target cell = %lu\n", i);
    serial::compute_plane_distances(
        i, target_node_offsets, target_cell_nodes, target_coordinates,
        candidate_offsets, candidates, source_node_offsets, source_cell_nodes,
        source_coordinates, distances,
        &n_plane_distances); // WILL DEFINITELY BREAK BECAUSE CAN'T JUST APPEND
                             // WITH THREADS
  }
  nvtxRangePop();

  ///////////////////////////////
  // parallel setup
  ///////////////////////////////

  int threadsPerBlock = 256;
  int blocksPerGrid = (n_cells + threadsPerBlock - 1) / threadsPerBlock;
  fprintf(
      stderr,
      "\nKernel launch parameters: blocksPerGrid=%d, threadsPerBlock=%d\n\n",
      blocksPerGrid, threadsPerBlock);

  double *point_buffer;
  size_t *cell_nodes, *cell_node_offsets;

  ///////////////////////////////
  // Kitsune unified memory
  ///////////////////////////////

  nvtxRangePush("Centroid: Kitsune managed memory");

  point_buffer = allocate_and_fill<double>(
      KITSUNE, 2 * (n_nodes + n_cells), "source coordinates",
      source_coordinates, 2 * n_nodes, PFKind);
  cell_node_offsets = allocate_and_fill<size_t>(
      KITSUNE, n_cells + 1, "source cell node offsets", source_node_offsets,
      n_cells + 1, PFKind);
  cell_nodes = allocate_and_fill<size_t>(KITSUNE, source_node_offsets[n_cells],
                                         "source cell nodes", source_cell_nodes,
                                         source_node_offsets[n_cells], PFKind);

  // compute centroids
  if constexpr (LOG_LEVEL > 0)
    printf("\nComputing parallel centroids with Cuda managed memory...\n");
  nvtxMark("Centroid Kitsune kernel start...");
  forall(size_t i = 0; i < n_cells; ++i)
      serial::centroid(i, cell_node_offsets, cell_nodes, point_buffer,
                       &point_buffer[2 * n_nodes]);
  nvtxMark("Centroid kernel end");

  // check centroids
  if constexpr (LOG_LEVEL > 0)
    for (size_t i = 0; i < n_cells; ++i)
      printf("cell %3zu centroid: %.4f, %.4f\n", i,
             point_buffer[2 * (i + n_nodes)],
             point_buffer[2 * (i + n_nodes) + 1]);
  if constexpr (CHECK)
    check_equal(&point_buffer[2 * n_nodes], source_centroids, 2 * n_cells);

  // cleanup
  //   __kitrt_cuMemFree(point_buffer);
  //   __kitrt_cuMemFree(cell_nodes);
  //   __kitrt_cuMemFree(cell_node_offsets);
  nvtxRangePop();

  //////////////////////////////////
  // create the mesh in UVM directly
  //////////////////////////////////

  nvtxRangePush("Mesh Creation in UVM");

  double *source_coordinates_uvm, *target_coordinates_uvm;
  size_t *source_cell_nodes_uvm, *source_node_offsets_uvm;
  size_t *target_cell_nodes_uvm, *target_node_offsets_uvm;
  size_t *candidates_uvm, *candidate_offsets_uvm;

  create_meshes(KITSUNE, nx, ny, x_max, y_max, shift_x, shift_y,
                source_coordinates_uvm, source_cell_nodes_uvm,
                source_node_offsets_uvm, target_coordinates_uvm,
                target_cell_nodes_uvm, target_node_offsets_uvm, candidates_uvm,
                candidate_offsets_uvm, shuffle, 2 * n_cells);

  nvtxRangePop();

  nvtxRangePush("Centroid on mesh defined with UVM");
  // compute centroids
  if constexpr (LOG_LEVEL > 0)
    printf("\nComputing parallel centroids with Cuda managed memory...\n");
  nvtxMark("Centroid Kitsune kernel start...");
  // raises a runtime KITSUNE WARNING if we use in the centroid kernel as the
  // final argument
  double *results = &source_coordinates_uvm[2 * n_nodes];
  forall(size_t i = 0; i < n_cells; ++i) serial::centroid(
      i, source_node_offsets_uvm, source_cell_nodes_uvm, source_coordinates_uvm,
      &source_coordinates_uvm[2 * n_nodes]);
  nvtxMark("Centroid kernel end");

  // check centroids
  if constexpr (LOG_LEVEL > 0)
    for (size_t i = 0; i < n_cells; ++i)
      printf("cell %3zu centroid: %.4f, %.4f\n", i, results[2 * i],
             results[2 * i + 1]);
  if constexpr (CHECK)
    check_equal(results, source_centroids, 2 * n_cells);

  nvtxRangePop();

  ///////////////////////////////
  // free application heap memory
  ///////////////////////////////

  // free heap memory
  free(source_coordinates);
  free(source_cell_nodes);
  free(source_node_offsets);
  free(target_coordinates);
  free(target_cell_nodes);
  free(target_node_offsets);
  free(candidates);
  free(candidate_offsets);
  free(source_centroids);

  return 0;
}
