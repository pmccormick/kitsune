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
#pragma once

#include <cstdio>
#include <string>
#include <vector>
#include <kitsune.h>
#include "kitrt/cuda.h"
#include "nvToolsExt.h"

// #include <kitsune.h>
// #include "kitsune/kitrt/llvm-gpu.h"
// #include "kitsune/kitrt/kitrt-cuda.h"

const int LOG_LEVEL = 0;

enum PrefetchKinds {
  EXPLICIT = 0,  // Use explicit async prefetch calls.
  PRELAUNCH = 1, // Prelaunch the kernel to move pages to device.
  NONE = 2       // Do nothing, default to built-in page management.
};

enum MemoryType {
  HOST = 0,    // Allocate on the host
  KITSUNE = 1 // Let Kitsune do the memory management
};

// parse the command line arguments
void parse_args(int argc, char *argv[], int *pnx, int *pny, bool *pshuffle,
                PrefetchKinds *pPFKind);

// row major, because why not?
inline size_t index(int ix, int iy, int nx);

// create the node coordinates of a rectangular structured mesh
void create_coordinates(MemoryType memory_type, double *&coordinates, int nx,
                        int ny, double x_max, double y_max, double shift_x,
                        double shift_y, const char *label,
                        std::vector<size_t> *shuffle_nodes, size_t extra_bytes);

// create the cell to node topology
void create_cell_nodes(MemoryType memory_type, size_t *&cell_nodes, int nx,
                       int ny, const char *label,
                       std::vector<size_t> *shuffle_nodes,
                       std::vector<size_t> *shuffle_cells);

// two pass algorithm so compact sparse data representation
void create_candidates(MemoryType memory_type, int nx, int ny,
                       size_t *&candidates, size_t *&offsets, const char *label,
                       std::vector<size_t> *shuffle_cells);

// create source and target mesh as well as intersection candidates
void create_meshes(MemoryType memory_type, int nx, int ny, double x_max,
                   double y_max, double shift_x, double shift_y,
                   double *&source_coordinates, size_t *&source_cell_nodes,
                   size_t *&source_node_offsets, double *&target_coordinates,
                   size_t *&target_cell_nodes, size_t *&target_node_offsets,
                   size_t *&candidates, size_t *&candidate_offsets,
                   bool shuffle, std::vector<size_t> &shuffle_source_nodes, 
                   std::vector<size_t> &shuffle_source_cells, std::vector<size_t> &shuffle_target_nodes, size_t extra_bytes = 0);




// create the node coordinates of a rectangular structured mesh
void create_coordinates_gpu(double *&coordinates, int nx,
                        int ny, double x_max, double y_max, double shift_x,
                        double shift_y, const char *label,
                        std::vector<size_t> *shuffle_nodes, size_t extra_bytes);

// create the cell to node topology
void create_cell_nodes_gpu(size_t *&cell_nodes, int nx,
                       int ny, const char *label,
                       std::vector<size_t> *shuffle_nodes,
                       std::vector<size_t> *shuffle_cells);

// return sizes so that the candidates can be allocated
size_t create_candidate_offsets(size_t *&offsets, int nx, int ny);

// two pass algorithm so compact sparse data representation
void create_candidates_gpu(size_t *&candidates, size_t n_candidates, int nx, int ny,
                       size_t *offsets, const char *label,
                       std::vector<size_t> *shuffle_cells);

// create source and target mesh as well as intersection candidates
void create_meshes_gpu(int nx, int ny, double x_max,
                   double y_max, double shift_x, double shift_y,
                   double *&source_coordinates, size_t *source_cell_nodes,
                   size_t *source_node_offsets, double *&target_coordinates,
                   size_t *target_cell_nodes, size_t *target_node_offsets,
                   /*size_t *&candidates,*/ size_t *candidate_offsets,
                   bool shuffle, std::vector<size_t> &shuffle_source_nodes, 
                   std::vector<size_t> &shuffle_source_cells, std::vector<size_t> &shuffle_target_nodes);




// check results
template <class T> int check_equal(T *v1, T *v2, size_t n) {
  size_t n_unequal = 0;
  for (size_t i = 0; i < n; ++i) {
    if (v1[i] != v2[i]) {
      if constexpr (LOG_LEVEL > 0) {
        if constexpr (std::is_same<T, double *>::value)
          printf("Vectors not equal at %lu: %f %f\n", i, v1[i], v2[i]);
        else if constexpr (std::is_same<T, size_t *>::value)
          printf("Vectors not equal at %lu: %lu %lu\n", i, v1[i], v2[i]);
      }
      ++n_unequal;
    }
  }
  if (n_unequal > 0)
    printf("%lu/%lu values were unequal.\n", n_unequal, n);
  else
    printf("All %lu values were equal!\n", n);
  return n_unequal > 0 ? 1 : 0;
}

template <class T>
T *allocate(MemoryType memory_type, size_t n, const char *label) {

  T *device_buffer;
  std::string message;

  switch (memory_type) {
  case HOST:
    message = std::string("malloc ") + label;
    nvtxMark(message.c_str());
    return (T *)malloc(sizeof(T) * n);
  case KITSUNE:
    message = std::string("__kitrt_cuMemAllocManaged ") + label;
    nvtxMark(message.c_str());
    // fprintf(stderr, "Allocating %s\n", message.c_str());
    return (T *)__kitrt_cuMemAllocManaged(sizeof(T) * n);
  }
  return nullptr;
}

template <class T>
void fill(MemoryType memory_type, T *device_buffer, T *host_buffer,
           size_t n_copy, const char *label, PrefetchKinds PFKind) {

  std::string message;

  switch (memory_type) {
  case HOST:
    message = std::string("memcopy ") + label;
    nvtxMark(message.c_str());
    memcpy(device_buffer, host_buffer, sizeof(T) * n_copy);
    break;
  case KITSUNE:
    message = std::string("memcopy ") + label + " to Kitsune managed";
    nvtxMark(message.c_str());
    memcpy(device_buffer, host_buffer, sizeof(T) * n_copy);
    break;
  }
}

template <class T>
T *allocate_and_fill(MemoryType memory_type, size_t n, const char *label,
                     T *host_buffer, size_t n_copy, PrefetchKinds PFKind) {

  T *device_buffer = allocate<T>(memory_type, n, label);
  fill<T>(memory_type, device_buffer, host_buffer, n_copy, label, PFKind);
  return device_buffer;
}
