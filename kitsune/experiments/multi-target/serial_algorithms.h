//
// Copyright(c) 2020 Triad National Security, LLC
// All rights reserved.
//
// This file is part of the kitsune / llvm project.    It is released under
// the LLVM license.
//
// Simple example of mesh intersection. Hopefully represents relevant memory
// access patterns.
#pragma once
namespace serial {

inline void centroid(size_t i, size_t *source_node_offsets,
                     size_t *source_cell_nodes, double *source_coordinates,
                     double *source_centroids) {
  double xc = 0;
  double yc = 0;
  size_t offset_start = source_node_offsets[i],
         offset_end = source_node_offsets[i + 1];
  for (size_t offset = offset_start; offset < offset_end; ++offset) {
    size_t source_node = source_cell_nodes[offset];
    xc += source_coordinates[2 * source_node];
    yc += source_coordinates[2 * source_node + 1];
  }
  source_centroids[2 * i] = xc / (offset_end - offset_start);
  source_centroids[2 * i + 1] = yc / (offset_end - offset_start);
}

inline void compute_plane_distances(
    const size_t i, const size_t *target_node_offsets,
    const size_t *target_cell_nodes, const double *target_coordinates,
    const size_t *candidate_offsets, const size_t *candidates,
    const size_t *source_node_offsets, const size_t *source_cell_nodes,
    const double *source_coordinates, double *distances,
    size_t *n_plane_distances) {
  // loop over nodes of the target cell
  for (size_t tni = target_node_offsets[i]; tni < target_node_offsets[i + 1];
       ++tni) {
    size_t tn_start = target_cell_nodes[tni],
           tn_end = target_cell_nodes[(tni == target_node_offsets[i + 1] - 1)
                                          ? target_node_offsets[i]
                                          : tni + 1];
    // compute the plane for the target segment defined by tn_start, tn_end
    // cells are counterclockwise
    double target_dx =
        target_coordinates[2 * tn_end] - target_coordinates[2 * tn_start];
    double target_dy = target_coordinates[2 * tn_end + 1] -
                       target_coordinates[2 * tn_start + 1];

    // normal is outward
    double nx = target_dy;
    double ny = -target_dx;

    // plane is defined by n \dot x -d =0
    double d = nx * target_coordinates[2 * tn_start] +
               ny * target_coordinates[2 * tn_start + 1];

    //   LOG_LEVEL > 1 &&
    //       printf("  target edge = (%lu, %lu), (%.2f, %.2f)->(%.2f, %.2f),
    //       n=(%.2f, %.2f), d=%.2f\n",
    //              tn_start, tn_end, target_coordinates[2 * tn_start],
    //              target_coordinates[2 * tn_start + 1],
    //              target_coordinates[2 * tn_end],
    //              target_coordinates[2 * tn_end + 1],
    //              nx,ny,d);

    // loop over neighboring source cells
    for (size_t cci = candidate_offsets[i]; cci < candidate_offsets[i + 1];
         ++cci) {
      size_t candidate = candidates[cci];
      // LOG_LEVEL>1 && printf("    candidate source cell = %lu, edges:\n ",
      // candidate);
      // LOG_LEVEL > 1 &&
      //     printf("    candidate source cell = %lu, nodes:\n      ",
      //            candidate);
      // source cell = candidate, target cell=i
      // we need to decompose either source or target or both into triangles
      // in order to guarantee convexity
      for (size_t sni = source_node_offsets[candidate];
           sni < source_node_offsets[candidate + 1]; ++sni) {
        size_t sn_start = source_cell_nodes[sni]; //,
        //   sn_end =
        //       source_cell_nodes[sni == source_node_offsets[candidate + 1] -
        //                                     1
        //                             ? source_node_offsets[candidate]
        //                             : sni + 1];
        // LOG_LEVEL>1 && printf("(%lu, %lu) ", sn_start, sn_end);
        //   LOG_LEVEL > 1 && printf("%lu (%.2f, %.2f) ", sn_start,
        //                           source_coordinates[2 * sn_start],
        //                           source_coordinates[2 * sn_start + 1]);

        // calculate the plane distance
        // source node, target start, end
        // every triangulated target edge (n1, n2), (n2,cen), (cen, n1)
        // computed against every candidate source node the number of times we
        // get here is basically the target nodes * source nodes for each pair
        // in the candiate list

        // double dist = plane_distance(source_coordinates[2*s])
        // size_t target_cell_centroid=n_nodes+i,
        // source_cell_centroid=n_nodes+sni; compute
        double dist = nx * source_coordinates[2 * sn_start] +
                      ny * source_coordinates[2 * sn_start + 1] - d;
        //   LOG_LEVEL > 1 && printf("d=%.2f    ", dist);
        distances[(*n_plane_distances)++] = dist;
      }
      // LOG_LEVEL > 1 && printf("\n");
    }
  }
  // printf("\n");
}
} // namespace serial
