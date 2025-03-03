#include "Grid.h"
#include <cmath>
#include <algorithm>

Grid::Grid(size_t nx, size_t ny, double width, double height)
  : nx_(nx), ny_(ny), width_(width), height_(height) {

  if (nx < 2 || ny < 2) {
    throw std::invalid_argument("Grid dimensions must be at least 2x2");
  }

  dx_ = width_ / (nx_ - 1);
  dy_ = height_ / (ny_ - 1);

  // Pre-allocate cells for performance
  cells_.resize(nx_ * ny_);

  // Mark boundary cells
  for (size_t i = 0; i < nx_; ++i) {
    getCell(i, 0).setBoundary(true);
    getCell(i, ny_-1).setBoundary(true);
  }

  for (size_t j = 0; j < ny_; ++j) {
    getCell(0, j).setBoundary(true);
    getCell(nx_-1, j).setBoundary(true);
  }
}

  // Set material for a specific region
void Grid::setMaterialRegion(size_t i_start, size_t i_end,
		       size_t j_start, size_t j_end,
		       Material* material) {
  if (!material) {
    throw std::invalid_argument("Material cannot be null");
  }

  i_end = std::min(i_end, nx_-1);
  j_end = std::min(j_end, ny_-1);

  for (size_t j = j_start; j <= j_end; ++j) {
    for (size_t i = i_start; i <= i_end; ++i) {
      getCell(i, j).setMaterial(material);
    }
  }
}

  // Define a circular obstacle
void Grid::setCircularObstacle(double center_x, double center_y, double radius, Material* material) {
  if (!material) {
    throw std::invalid_argument("Material cannot be null");
  }

  // Convert physical coordinates to grid indices
  size_t center_i = gridI(center_x);
  size_t center_j = gridJ(center_y);
  size_t radius_cells = static_cast<size_t>(radius / std::min(dx_, dy_) + 0.5);

  // Set cells within radius as obstacles with the specified material
  for (size_t j = std::max(center_j, radius_cells) - radius_cells;
       j <= std::min(center_j + radius_cells, ny_-1); ++j) {
    for (size_t i = std::max(center_i, radius_cells) - radius_cells;
	 i <= std::min(center_i + radius_cells, nx_-1); ++i) {

      double dist_sq =
	std::pow((static_cast<double>(i) - center_i) * dx_, 2) +
	std::pow((static_cast<double>(j) - center_j) * dy_, 2);

      if (dist_sq <= radius * radius) {
	Cell& cell = getCell(i, j);
	cell.setObstacle(true);
	cell.setMaterial(material);
	cell.setVelocityX(0.0);  // No-slip condition for obstacles
	cell.setVelocityY(0.0);
      }
    }
  }
}


// Get velocity field for visualization or analysis
void Grid::getVelocityField(std::vector<double>& vx, std::vector<double>& vy) const {
  vx.resize(nx_ * ny_);
  vy.resize(nx_ * ny_);

  for (size_t j = 0; j < ny_; ++j) {
    for (size_t i = 0; i < nx_; ++i) {
      size_t idx = index(i, j);
      const Cell& cell = cells_[idx];
      vx[idx] = cell.getVelocityX();
      vy[idx] = cell.getVelocityY();
    }
  }
}

  // Get pressure field for visualization or analysis
void Grid::getPressureField(std::vector<double>& pressure) const {
  pressure.resize(nx_ * ny_);

  for (size_t j = 0; j < ny_; ++j) {
    for (size_t i = 0; i < nx_; ++i) {
      pressure[index(i, j)] = cells_[index(i, j)].getPressure();
    }
  }
}

  // Calculate divergence at a cell (useful for pressure solvers)
double Grid::calculateDivergence(size_t i, size_t j) const {
  if (i == 0 || i == nx_-1 || j == 0 || j == ny_-1) {
    return 0.0;  // Zero divergence at boundaries
  }

  const double vx_right = getCell(i+1, j).getVelocityX();
  const double vx_left = getCell(i-1, j).getVelocityX();
  const double vy_top = getCell(i, j+1).getVelocityY();
  const double vy_bottom = getCell(i, j-1).getVelocityY();

  return (vx_right - vx_left) / (2.0 * dx_) +
    (vy_top - vy_bottom) / (2.0 * dy_);
}

