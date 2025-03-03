#include <vector>
#include <memory>
#include <algorithm>
#include <stdexcept>
#include <cmath>

// Assuming these are already defined in the project
#include "Cell.h"
#include "Material.h"

class Grid {
 private:
  // Grid dimensions
  size_t nx_;
  size_t ny_;

  // Physical domain size
  double width_;
  double height_;

  // Cell size
  double dx_;
  double dy_;

  // Storage for cells
  std::vector<Cell> cells_;

 public:
  Grid(size_t nx, size_t ny, double width, double height);

  // Index calculation for 1D array - inline for performance
  inline size_t index(size_t i, size_t j) const {
    return i + j * nx_;
  }

  // Cell access methods with bounds checking in debug builds
  inline Cell& getCell(size_t i, size_t j) {
#ifdef DEBUG
    if (i >= nx_ || j >= ny_) {
      throw std::out_of_range("Cell indices out of range");
    }
#endif
    return cells_[index(i, j)];
  }

  const Cell& getCell(size_t i, size_t j) const {
#ifdef DEBUG
    if (i >= nx_ || j >= ny_) {
      throw std::out_of_range("Cell indices out of range");
    }
#endif
    return cells_[index(i, j)];
  }

  // Direct array access for high-performance loops
  inline Cell* getCellData() { return cells_.data(); }
  inline const Cell* getCellData() const { return cells_.data(); }

  // Grid properties
  inline size_t getNx() const { return nx_; }
  inline size_t getNy() const { return ny_; }
  inline double getDx() const { return dx_; }
  inline double getDy() const { return dy_; }
  inline double getWidth() const { return width_; }
  inline double getHeight() const { return height_; }

  // Convert from grid indices to physical coordinates
  inline double physicalX(size_t i) const { return i * dx_; }
  inline double physicalY(size_t j) const { return j * dy_; }

  // Convert from physical coordinates to grid indices
  inline size_t gridI(double x) const {
    size_t i = static_cast<size_t>(x / dx_ + 0.5);
    return std::min(i, nx_-1);  // Clamp to valid range
  }

  inline size_t gridJ(double y) const {
    size_t j = static_cast<size_t>(y / dy_ + 0.5);
    return std::min(j, ny_-1);  // Clamp to valid range
  }

  // Initialize grid with default material
  inline void initialize(Material* default_material) {
    if (!default_material) {
      throw std::invalid_argument("Default material cannot be null");
    }

    for (auto& cell : cells_) {
      cell.setMaterial(default_material);
    }
  }

  // Set material for a specific region
  void setMaterialRegion(size_t i_start, size_t i_end,
			 size_t j_start, size_t j_end,
			 Material* material);

  // Define a circular obstacle
  void setCircularObstacle(double center_x, double center_y, double radius, Material* material);

  // Clear velocities across the grid (useful between simulation steps)
  inline void clearVelocities() {
    for (auto& cell : cells_) {
      if (!cell.isObstacle()) {  // Preserve zero velocity for obstacles
	cell.setVelocityX(0.0);
	cell.setVelocityY(0.0);
      }
    }
  }

  // Clear pressures across the grid
  inline void clearPressures() {
    for (auto& cell : cells_) {
      cell.setPressure(0.0);
    }
  }

  // Utility methods for high-performance solvers

  // Get velocity field for visualization or analysis
  void getVelocityField(std::vector<double>& vx, std::vector<double>& vy) const;

  // Get pressure field for visualization or analysis
  void getPressureField(std::vector<double>& pressure) const;

  // Calculate divergence at a cell (useful for pressure solvers)
  double calculateDivergence(size_t i, size_t j) const;

  // Check if grid is properly initialized
  bool isInitialized() const {
    for (const auto& cell : cells_) {
      if (cell.getMaterial() == nullptr) {
	return false;
      }
    }
    return true;
  }
};
