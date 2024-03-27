//
// Copyright(c) 2020 Triad National Security, LLC
// All rights reserved.
//
// This file is part of the kitsune / llvm project.    It is released under
// the LLVM license.
//
// Simple example of mesh intersection. Hopefully represents relevant memory
// access patterns.

#include <algorithm>
#include <chrono>
#include <random>

#include "common.h"
#include <kitsune.h>
// #include "kitrt/cuda.h"

using namespace std;

void parse_args(int argc, char *argv[], int *pnx, int *pny, bool *pshuffle,
                PrefetchKinds *pPFKind) {

  // handle command lines
  if (argc > 1)
    *pnx = atoi(argv[1]);
  if (argc > 2)
    *pny = atoi(argv[2]);
  if (argc > 3)
    if (std::string(argv[3]) == "shuffle")
      *pshuffle = true;
  if (argc > 5) {
    if (std::string(argv[4]) == "explicit")
      *pPFKind = EXPLICIT;
    else if (std::string(argv[4]) == "pre-launch")
      *pPFKind = PRELAUNCH;
    else
      *pPFKind = NONE;
  }

  // diagnostics
  // if constexpr (LOG_LEVEL > 0)
  printf("\nnx=%d, ny=%d\nCells were %sshuffled\n", *pnx, *pny,
         *pshuffle ? "" : "not ");
  printf("Prefetch kind = %s\n", *pPFKind == 0   ? "explicit"
                                 : *pPFKind == 1 ? "pre-launch"
                                                 : "none");
}

// row major, because why not?
inline size_t index(int ix, int iy, int nx) { return iy * nx + ix; }
inline void index2ij(const size_t i, const int nx, int &ix, int&iy){
    ix = i % nx;
    iy = i / nx;
}

//////////////////////////////////////////////////////////////
// loop over i,j variants
//////////////////////////////////////////////////////////////

// create the node coordinates of a rectangular structured mesh
void create_coordinates(MemoryType memory_type, double *&coordinates, int nx,
                        int ny, double x_max, double y_max, double shift_x,
                        double shift_y, const char *label = "",
                        std::vector<size_t> *shuffle_nodes = nullptr, size_t extra_elements=0) {

  const double dx = x_max / nx, dy = y_max / ny;
  coordinates = allocate<double>(memory_type, 2 * (nx + 1) * (ny + 1) + extra_elements, label);

  std::string message = std::string("create_coordinates ") + label;
  nvtxMark(message.c_str());

  // create coordinates
  // i,j are the unshuffled indices
  size_t unshuffled_node_id = 0;
  for (int j = 0; j < ny + 1; ++j)
    for (int i = 0; i < nx + 1; ++i) {
      size_t node_id = shuffle_nodes ? (*shuffle_nodes)[unshuffled_node_id]
                                     : unshuffled_node_id;
      coordinates[2 * node_id] = dx * i + shift_x;
      coordinates[2 * node_id + 1] = dy * j + shift_y;

      if constexpr (LOG_LEVEL > 0)
        printf("node=%3lu, unshuffled=%3lu, x=%f, y=%f\n", node_id,
               unshuffled_node_id, coordinates[2 * node_id],
               coordinates[2 * node_id + 1]);

      ++unshuffled_node_id;
    }
}

// create the cell to node topology
void create_cell_nodes(MemoryType memory_type, size_t *&nodes, int nx, int ny,
                       const char *label,
                       std::vector<size_t> *shuffle_nodes = nullptr,
                       std::vector<size_t> *shuffle_cells = nullptr) {

  // allocate coordinates
  nodes = allocate<size_t>(memory_type, 4 * nx * ny, label);

  // create cell to node mapping, just iterating the stencil explicitly
  if constexpr (LOG_LEVEL > 0)
    printf("\n");
  std::string message = std::string("create_cell_nodes ") + label;
  nvtxMark(message.c_str());

  size_t unshuffled_cell_id = 0;
  // i,j are the unshuffled cell indices
  for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i) {
      size_t cell_id = shuffle_cells ? (*shuffle_cells)[unshuffled_cell_id]
                                     : unshuffled_cell_id;
      if (shuffle_nodes) {
        nodes[4 * cell_id] = (*shuffle_nodes)[j * (nx + 1) + i];
        nodes[4 * cell_id + 1] = (*shuffle_nodes)[j * (nx + 1) + i + 1];
        nodes[4 * cell_id + 2] = (*shuffle_nodes)[(j + 1) * (nx + 1) + i + 1];
        nodes[4 * cell_id + 3] = (*shuffle_nodes)[(j + 1) * (nx + 1) + i];
      } else {
        nodes[4 * cell_id] = j * (nx + 1) + i;
        nodes[4 * cell_id + 1] = j * (nx + 1) + i + 1;
        nodes[4 * cell_id + 2] = (j + 1) * (nx + 1) + i + 1;
        nodes[4 * cell_id + 3] = (j + 1) * (nx + 1) + i;
      }
      if constexpr (LOG_LEVEL > 0)
        printf("cell=%3lu, unshuffled=%3lu,    nodes: %3lu %3lu %3lu %3lu\n",
               cell_id, unshuffled_cell_id, nodes[4 * cell_id],
               nodes[4 * cell_id + 1], nodes[4 * cell_id + 2],
               nodes[4 * cell_id + 3]);

      ++unshuffled_cell_id;
    }
}

// two pass algorithm so compact sparse data representation
void create_candidates(MemoryType memory_type, int nx, int ny,
                       size_t *&candidates, size_t *&offsets, const char *label,
                       std::vector<size_t> *shuffle_cells = nullptr) {

  int n_candidates = 0;

  // loop once over cells for sizing
  // for calculating the size , we don't need to worry about shuffling
  for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i)
      // loop over stencil
      for (int j_neighbor = j - 1; j_neighbor < j + 2; ++j_neighbor)
        for (int i_neighbor = i - 1; i_neighbor < i + 2; ++i_neighbor)
          if (i_neighbor >= 0 && i_neighbor < nx && j_neighbor >= 0 &&
              j_neighbor < ny)
            ++n_candidates;

  if constexpr (LOG_LEVEL > 0)
    printf("Expected candidates: %d, Total candidates: %d\n",
           9 * (nx - 2) * (ny - 2) + 6 * 2 * (nx - 2 + ny - 2) + 4 * 4,
           n_candidates);

  // allocate the compact sparse candidates
  offsets = allocate<size_t>(memory_type, nx * ny + 1,
                             (std::string("offsets for ") + label).c_str());
  candidates = allocate<size_t>(memory_type, n_candidates, label);
  std::string message = std::string("create_candidates ") + label + " , and offsets";
  nvtxMark(message.c_str());


  // loop again over cells to actually assign the candidates
  size_t target_cell_id = 0, offset = 0;
  for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i) {

      offsets[target_cell_id++] = offset;

      // loop over stencil, only includes cells that are indexed in the domain
      for (int j_neighbor = j - 1; j_neighbor < j + 2; ++j_neighbor)
        for (int i_neighbor = i - 1; i_neighbor < i + 2; ++i_neighbor)
          if (i_neighbor >= 0 && i_neighbor < nx && j_neighbor >= 0 &&
              j_neighbor < ny)
            candidates[offset++] =
                shuffle_cells
                    ? (*shuffle_cells)[index(i_neighbor, j_neighbor, nx)]
                    : index(i_neighbor, j_neighbor, nx);
    }

  // set the final offset to the final element of the array
  offsets[target_cell_id] = n_candidates;

  if constexpr (LOG_LEVEL > 0)
    for (size_t id = 0; id < nx * ny; ++id) {
      printf("target cell id=%lu offset=%lu source candidates: ", id,
             offsets[id]);
      for (size_t i = offsets[id]; i < offsets[id + 1]; ++i)
        printf("%lu ", candidates[i]);
      printf("\n");
    }
}

void create_meshes(MemoryType memory_type, int nx, int ny, double x_max,
                   double y_max, double shift_x, double shift_y,
                   double *&source_coordinates, size_t *&source_cell_nodes,
                   size_t *&source_node_offsets, double *&target_coordinates,
                   size_t *&target_cell_nodes, size_t *&target_node_offsets,
                   size_t *&candidates, size_t *&candidate_offsets,
                   bool shuffle, std::vector<size_t> &shuffle_source_nodes, 
                   std::vector<size_t> &shuffle_source_cells, std::vector<size_t> &shuffle_target_nodes, size_t extra_elements) {

  ////////////////////////////////////////////
  // create the meshes
  ///////////////////////////////////////////

  // create source mesh
  if constexpr (LOG_LEVEL > 0)
    printf("\nSource Mesh:\n");
  create_coordinates(memory_type, source_coordinates, nx, ny, x_max, y_max, 0., 0.,
                     "source_coordinates",
                     shuffle ? &shuffle_source_nodes : nullptr, extra_elements);
  create_cell_nodes(memory_type, source_cell_nodes, nx, ny, "source_cell_nodes",
                    shuffle ? &shuffle_source_nodes : nullptr,
                    shuffle ? &shuffle_source_cells : nullptr);
  source_node_offsets = allocate<size_t>(
      memory_type, nx * ny + 1, "source_node_offsets"); // just fake since regular
  for (size_t i = 0; i < nx * ny + 1; ++i)
    source_node_offsets[i] = 4 * i;

  // create target mesh
  if constexpr (LOG_LEVEL > 0)
    printf("\nTarget Mesh:\n");
  create_coordinates(memory_type, target_coordinates, nx, ny, x_max, y_max, shift_x,
                     shift_y, "target_coordinates",
                     shuffle ? &shuffle_target_nodes : nullptr, extra_elements);
  // don't need shuffle target cells
  create_cell_nodes(memory_type, target_cell_nodes, nx, ny, "target_cell_nodes",
                    shuffle ? &shuffle_target_nodes : nullptr); 
  target_node_offsets = allocate<size_t>(
      memory_type, (nx * ny + 1), "target_node_offsets"); // just fake since regular
  for (size_t i = 0; i < nx * ny + 1; ++i)
    target_node_offsets[i] = 4 * i;

  // create candidates and offsets
  if constexpr (LOG_LEVEL > 0)
    printf("\nIntersection Candidates:\n");
  create_candidates(memory_type, nx, ny, candidates, candidate_offsets,
                    "create candidate ",
                    shuffle ? &shuffle_source_cells : nullptr);
}



//////////////////////////////////////////////////////////////
// single loop index variants
//////////////////////////////////////////////////////////////



// create the node coordinates of a rectangular structured mesh
void create_coordinates_gpu(double *&coordinates, int nx,
                        int ny, double x_max, double y_max, double shift_x,
                        double shift_y, const char *label = "",
                        std::vector<size_t> *shuffle_nodes = nullptr, size_t extra_elements=0) {

  const double dx = x_max / nx, dy = y_max / ny;
  const size_t n_nodes = (nx + 1) * (ny + 1);
  
  coordinates = allocate<double>(KITSUNE, 2 * n_nodes + extra_elements, label);

  std::string message = std::string("create_coordinates_gpu ") + label;
  nvtxMark(message.c_str());

  // create coordinates
  forall (size_t n = 0; n < n_nodes; ++n){
    int i,j;
    index2ij(n, nx+1, i, j);
    size_t node_id = shuffle_nodes ? (*shuffle_nodes)[n] : n;
    coordinates[2 * node_id] = dx * i + shift_x;
    coordinates[2 * node_id + 1] = dy * j + shift_y;

    if constexpr (LOG_LEVEL > 0)
    printf("node=%3lu, unshuffled=%3lu, x=%f, y=%f, %s\n", node_id,
            n, coordinates[2 * node_id],
            coordinates[2 * node_id + 1], label);
  }
}

// create the cell to node topology
void create_cell_nodes_gpu(size_t *&nodes, int nx, int ny,
                       const char *label,
                       std::vector<size_t> *shuffle_nodes = nullptr,
                       std::vector<size_t> *shuffle_cells = nullptr) {

  nodes = allocate<size_t>(KITSUNE, 4 * nx * ny, "source cell nodes");

  // create cell to node mapping, just iterating the stencil explicitly
  if constexpr (LOG_LEVEL > 0)
    printf("\n");
  std::string message = std::string("create_cell_nodes_gpu ") + label;
  nvtxMark(message.c_str());

  const size_t n_cells = nx * ny;

  forall (size_t c = 0; c < n_cells; ++c){
    int i,j;
    index2ij(c, nx, i, j);
    size_t cell_id = shuffle_cells ? (*shuffle_cells)[c]
                                    : c;
    if (shuffle_nodes) {
    nodes[4 * cell_id] = (*shuffle_nodes)[j * (nx + 1) + i];
    nodes[4 * cell_id + 1] = (*shuffle_nodes)[j * (nx + 1) + i + 1];
    nodes[4 * cell_id + 2] = (*shuffle_nodes)[(j + 1) * (nx + 1) + i + 1];
    nodes[4 * cell_id + 3] = (*shuffle_nodes)[(j + 1) * (nx + 1) + i];
    } else {
    nodes[4 * cell_id] = j * (nx + 1) + i;
    nodes[4 * cell_id + 1] = j * (nx + 1) + i + 1;
    nodes[4 * cell_id + 2] = (j + 1) * (nx + 1) + i + 1;
    nodes[4 * cell_id + 3] = (j + 1) * (nx + 1) + i;
    }
    if constexpr (LOG_LEVEL > 0)
    printf("cell=%3lu, unshuffled=%3lu,    nodes: %3lu %3lu %3lu %3lu, %s\n",
            cell_id, c, nodes[4 * cell_id],
            nodes[4 * cell_id + 1], nodes[4 * cell_id + 2],
            nodes[4 * cell_id + 3], label);
  }
}


// find candidate sizes
size_t create_candidate_offsets(size_t *&offsets, const int nx, const int ny){

  offsets  = allocate<size_t>(KITSUNE, nx * ny + 1,"offsets for candidates"); 

  if constexpr (LOG_LEVEL > 0)
    printf("\n");
  std::string message = std::string("create_candidate_offsets ");
  nvtxMark(message.c_str());

  const size_t n_cells = nx * ny;

  // declare and initialize the candidate counts
  // I think we could optimize this away and do in place in offsets
  // but for the moment it isn't worth the effort
  int *candidate_sizes= (int*) malloc (sizeof(int)*n_cells);
//   int candidate_sizes[n_cells];
  for (int i = 0; i < n_cells; ++i) candidate_sizes[i]=0;

  // parallel loop over target cells
  // Target cells aren't shuffled because we just loop over target cells anyway.
  // Target cells are in canonical position so we can trivially determine how
  // many neighbors they should have: 4 in the corners, 6 on edges, 9 on the interior.
  forall (size_t c = 0; c < n_cells; ++c){
    int i,j;
    index2ij(c, nx, i, j);
    // loop over stencil
    for (int j_neighbor = j - 1; j_neighbor < j + 2; ++j_neighbor)
      for (int i_neighbor = i - 1; i_neighbor < i + 2; ++i_neighbor)
        if (i_neighbor >= 0 && i_neighbor < nx && j_neighbor >= 0 &&
            j_neighbor < ny)
          candidate_sizes[c]++;
  }

  // serial partial sum loop to compute offsets
  offsets[0]=0;
  for (size_t c = 0; c < n_cells; ++c){
    offsets[c+1]=offsets[c] + candidate_sizes[c];
  } 

  if constexpr (LOG_LEVEL > 0)
    printf("Expected candidates: %d, Total candidates: %lu\n",
           9 * (nx - 2) * (ny - 2) + 6 * 2 * (nx - 2 + ny - 2) + 4 * 4,
           offsets[n_cells]);
  free (candidate_sizes);
  return offsets[n_cells];
}


// two pass algorithm so compact sparse data representation
void create_candidates_gpu(size_t *&candidates, size_t n_candidates, int nx, int ny,
                       size_t *offsets, const char *label,
                       std::vector<size_t> *shuffle_cells = nullptr) {

  candidates = allocate<size_t>(KITSUNE, n_candidates, "create candidates");

  std::string message = std::string("create_candidates_gpu ") + label + " , and offsets";
  nvtxMark(message.c_str());


  // loop again over cells to actually assign the candidates
  size_t target_cell_id = 0, offset = 0;
  for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i) {

      // loop over stencil, only includes cells that are indexed in the domain
      for (int j_neighbor = j - 1; j_neighbor < j + 2; ++j_neighbor)
        for (int i_neighbor = i - 1; i_neighbor < i + 2; ++i_neighbor)
          if (i_neighbor >= 0 && i_neighbor < nx && j_neighbor >= 0 &&
              j_neighbor < ny)
            candidates[offset++] =
                shuffle_cells
                    ? (*shuffle_cells)[index(i_neighbor, j_neighbor, nx)]
                    : index(i_neighbor, j_neighbor, nx);
    }

  if constexpr (LOG_LEVEL > 0){
    printf("\nIntersection Candidates:\n");
    for (size_t id = 0; id < nx * ny; ++id) {
      printf("target cell id=%lu offset=%lu source candidates: ", id,
             offsets[id]);
      for (size_t i = offsets[id]; i < offsets[id + 1]; ++i)
        printf("%lu ", candidates[i]);
      printf("\n");
    }
  }
}

// // ATM candidates isn't used because we can't kitsune/cuda allocate inside a spawn
// // We need to create candidate_offsets then use the result of this to allocate 
// // candidates. I'm not deleting for now...
// void create_meshes_gpu(int nx, int ny, double x_max,
//                    double y_max, double shift_x, double shift_y,
//                    double *&source_coordinates, size_t *source_cell_nodes,
//                    size_t *source_node_offsets, double *&target_coordinates,
//                    size_t *target_cell_nodes, size_t *target_node_offsets,
//                    /*size_t *&candidates,*/ size_t *candidate_offsets,
//                    bool shuffle, std::vector<size_t> &shuffle_source_nodes, 
//                    std::vector<size_t> &shuffle_source_cells, std::vector<size_t> &shuffle_target_nodes) {


//   ////////////////////////////////////////////
//   // create the meshes
//   ///////////////////////////////////////////

//   nvtxMark("Beginning source and target mesh creation");

// //   // create source mesh
// //   if constexpr (LOG_LEVEL > 0)
// //     printf("\nSource Mesh:\n");


// //   spawn source_coordinates 
// //   {
// //   create_coordinates_gpu(source_coordinates, nx, ny, x_max, y_max, 0., 0.,
// //                      "source_coordinates",
// //                      shuffle ? &shuffle_source_nodes : nullptr,
// //                      2 * nx * ny);
// //   }

// //   spawn source_cell_nodes 
// //   {
// //   create_cell_nodes_gpu(source_cell_nodes, nx, ny, "source_cell_nodes",
// //                     shuffle ? &shuffle_source_nodes : nullptr,
// //                     shuffle ? &shuffle_source_cells : nullptr);
// //   }


// //   // create target mesh
// //   if constexpr (LOG_LEVEL > 0)
// //     printf("\nTarget Mesh:\n");

// //   spawn target_coordinates
// //   {
// //   create_coordinates_gpu(target_coordinates, nx, ny, x_max, y_max, shift_x,
// //                      shift_y, "target_coordinates",
// //                      shuffle ? &shuffle_target_nodes : nullptr,
// //                      2 * nx * ny);
// //   }
  
// //   spawn target_cell_nodes
// //   {
// //   // no need to shuffle the target cells
// //   create_cell_nodes_gpu(target_cell_nodes, nx, ny, "target_cell_nodes",
// //                     shuffle ? &shuffle_target_nodes : nullptr);
// //   }

// //   spawn source_node_offsets {
// //   forall (size_t i = 0; i < nx * ny + 1; ++i)
// //     source_node_offsets[i] = 4 * i;
// //   }

// //   spawn target_node_offsets {
// //   forall (size_t i = 0; i < nx * ny + 1; ++i)
// //     target_node_offsets[i] = 4 * i;
// //   }

// //   spawn create_candidate_offsets {
// //   create_candidate_offsets( nx, ny, candidate_offsets);
// //   }

//   // sync all the parallel work
// //   sync source_coordinates;
//   sync source_cell_nodes;
//   sync source_node_offsets;

// //   sync target_coordinates;
//   sync target_cell_nodes;
//   sync target_node_offsets;

//   sync create_candidate_offsets;

//   nvtxMark("After opencilk sync's");

//   return;
// }

