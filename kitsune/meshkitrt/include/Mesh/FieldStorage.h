#ifndef FIELD_STORAGE_H
#define FIELD_STORAGE_H

#include "Mesh.h"
#include <cstdint>
#include <cassert>
#include <cmath>

/**
 * @file FieldStorage.h
 * @brief Free-function based indexing strategies for field storage.
 *
 * This header encapsulates the field::storage namespace, which provides a collection
 * of free functions to compute the linear index from 2D grid coordinates. These functions
 * are fully inline and documented in detail.
 *
 * DESIGN NOTES:
 * - These functions enable a free-function approach to computing indices without wrapping
 *   them in an extra class.
 * - They are grouped under the field::storage namespace to avoid naming conflicts and to
 *   clearly separate layout functionality from other aspects of the Field class.
 * - The default implementation (rowMajor) is provided; additional functions (e.g., blocked,
 *   zOrder) can be added as needed.
 *
 * @see mesh::Mesh, Field.h
 */
namespace field {
  namespace storage {

    /**
     * @brief Computes the linear index for a row-major layout.
     *
     * The row-major ordering maps 2D indices (i, j) to a 1D index using the formula:
     *
     *     index = i + j * nx
     *
     * where nx is the number of cells in the x-direction.
     *
     * @param mesh Reference to the mesh providing grid dimensions.
     * @param i Column index.
     * @param j Row index.
     * @return int Linear index computed in row-major order.
     */
    static inline int rowMajor(const mesh::Mesh &mesh, int i, int j) {
      return i + j * mesh.nx();
    }

    /**
     * @brief Computes the linear index for a blocked layout.
     *
     * In a blocked layout, the grid is divided into blocks of fixed size. This example
     * assumes a block size that is a compile-time constant. The index is computed by first
     * determining the block position and then the offset within the block.
     *
     * @param mesh Reference to the mesh providing grid dimensions.
     * @param i Column index.
     * @param j Row index.
     * @param blockSize The size (in cells) of each block (assumed square).
     * @return int Linear index computed for a blocked layout.
     *
     * @note This function is provided as an example. In a production system, blockSize
     *       might be a template parameter or determined by code generation.
     */
    static inline int blocked(const mesh::Mesh &mesh, int i, int j, int blockSize) {
      // Determine block coordinates.
      int blocksPerRow = (mesh.nx() + blockSize - 1) / blockSize;
      int block_i = i / blockSize;
      int block_j = j / blockSize;

      // Determine local indices within the block.
      int local_i = i % blockSize;
      int local_j = j % blockSize;

      // Compute linear index.
      int index = (block_j * blocksPerRow + block_i) * (blockSize * blockSize)
	+ local_j * blockSize + local_i;
      return index;
    }

    /**
     * @brief Computes the linear index using a Z‑order (Morton) curve.
     *
     * Z‑order curves preserve locality and are sometimes used in spatial indexing.
     * This function interleaves the bits of i and j to compute the Morton code.
     *
     * @param mesh Reference to the mesh (only nx is used for range checking).
     * @param i Column index.
     * @param j Row index.
     * @return int Linear index computed via Z‑order curve.
     *
     * @note This function assumes that i and j are within the valid range and that
     *       the grid dimensions are powers of two. Additional error checking may be added.
     */
    static inline int zOrder(const mesh::Mesh &mesh, int i, int j) {
      auto interleaveBits = [](unsigned int x) -> unsigned int {
	x = (x | (x << 8)) & 0x00FF00FF;
	x = (x | (x << 4)) & 0x0F0F0F0F;
	x = (x | (x << 2)) & 0x33333333;
	x = (x | (x << 1)) & 0x55555555;
	return x;
      };

      unsigned int morton = (interleaveBits(j) << 1) | interleaveBits(i);
      return static_cast<int>(morton);
    }

    /**
     * @brief Default indexing function that wraps the row-major implementation.
     *
     * This function can be used by the Field class to abstract the indexing mechanism.
     * It calls rowMajor by default, but can be replaced by alternative functions (e.g., blocked,
     * zOrder) based on the mesh's storage policy.
     *
     * @param mesh Reference to the mesh.
     * @param i Column index.
     * @param j Row index.
     * @return int Linear index.
     */
    static inline int index(const mesh::Mesh &mesh, int i, int j) {
      // Default: use row-major layout.
      return rowMajor(mesh, i, j);
    }

  } // namespace storage
} // namespace field

#endif // FIELD_STORAGE_H
