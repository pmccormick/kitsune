/**
 * Implementation of Mesh class
 */
#include "Mesh.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

// Constructor
Mesh::Mesh(size_t nx, size_t ny, double width, double height, double origin_x,
           double origin_y)
    : m_nx(nx), m_ny(ny), m_width(width), m_height(height),
      m_origin_x(origin_x), m_origin_y(origin_y),
      m_dx(width / std::max(nx - 1, size_t(1))),
      m_dy(height / std::max(ny - 1, size_t(1))) {

  // Validate inputs
  if (nx < 2 || ny < 2) {
    throw std::invalid_argument("Mesh dimensions must be at least 2x2");
  }
  if (width <= 0.0 || height <= 0.0) {
    throw std::invalid_argument("Mesh dimensions must be positive");
  }

  // Allocate memory for all cells
  m_cells.resize(nx * ny);

  // Initialize cells and set grid references
  for (size_t j = 0; j < ny; ++j) {
    for (size_t i = 0; i < nx; ++i) {
      Cell &cell = getCell(i, j);

      // Set boundary cells at the domain perimeter
      if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) {
        cell.setType(Cell::CellType::BOUNDARY);
        cell.setBoundary(true);
        cell.setFixed(true);
      } else {
        cell.setType(Cell::CellType::FLUID);
      }
    }
  }
}

// Create a mesh with dimensions specified in non-SI units
std::shared_ptr<Mesh> Mesh::createWithUnits(size_t nx, size_t ny, double width,
                                            double height,
                                            const std::string &lengthUnit,
                                            double origin_x, double origin_y) {

  // Convert all dimensions to SI (meters)
  double widthMeters = Units::convert(width, lengthUnit, "m");
  double heightMeters = Units::convert(height, lengthUnit, "m");
  double originXMeters = Units::convert(origin_x, lengthUnit, "m");
  double originYMeters = Units::convert(origin_y, lengthUnit, "m");

  return std::make_shared<Mesh>(nx, ny, widthMeters, heightMeters,
                                originXMeters, originYMeters);
}

// Destructor
Mesh::~Mesh() {
  // No special cleanup needed as STL containers handle their own memory
}

// Cell access methods
Cell &Mesh::getCell(size_t i, size_t j) {
  if (i >= m_nx || j >= m_ny) {
    throw std::out_of_range("Cell indices out of range");
  }
  return m_cells[index(i, j)];
}

const Cell &Mesh::getCell(size_t i, size_t j) const {
  if (i >= m_nx || j >= m_ny) {
    throw std::out_of_range("Cell indices out of range");
  }
  return m_cells[index(i, j)];
}

// Initialize the mesh with a default material
void Mesh::initialize(std::shared_ptr<Material> defaultMaterial) {
  if (!defaultMaterial) {
    throw std::invalid_argument("Default material cannot be null");
  }

  for (Cell &cell : m_cells) {
    cell.setMaterial(defaultMaterial);
  }
}

// Set boundary condition for a standard boundary location
void Mesh::setBoundaryCondition(BoundaryLocation location,
                                std::shared_ptr<BoundaryClass> boundary) {

  if (!boundary) {
    throw std::invalid_argument("Boundary condition cannot be null");
  }

  // Get name of the boundary for map storage
  std::string boundaryName = boundary->getName();
  if (boundaryName.empty()) {
    // Generate a unique name if not provided
    static int counter = 0;
    boundaryName = "Boundary_" + std::to_string(counter++);
  }

  // Store in boundary map
  m_boundaryConditions[boundaryName] = boundary;

  // Apply to cells based on location
  std::vector<Cell *> boundaryCells = getBoundaryCells(location);

  for (Cell *cell : boundaryCells) {
    cell->setBoundaryCondition(boundary);
  }
}

// Set boundary condition for a custom region
void Mesh::setBoundaryConditionRegion(size_t i_start, size_t i_end,
                                      size_t j_start, size_t j_end,
                                      std::shared_ptr<BoundaryClass> boundary) {

  if (!boundary) {
    throw std::invalid_argument("Boundary condition cannot be null");
  }

  // Get name of the boundary for map storage
  std::string boundaryName = boundary->getName();
  if (boundaryName.empty()) {
    // Generate a unique name if not provided
    static int counter = 0;
    boundaryName = "Boundary_" + std::to_string(counter++);
  }

  // Store in boundary map
  m_boundaryConditions[boundaryName] = boundary;

  // Clamp indices to valid range
  i_start = std::min(i_start, m_nx - 1);
  i_end = std::min(i_end, m_nx - 1);
  j_start = std::min(j_start, m_ny - 1);
  j_end = std::min(j_end, m_ny - 1);

  // Apply to cells in the region
  for (size_t j = j_start; j <= j_end; ++j) {
    for (size_t i = i_start; i <= i_end; ++i) {
      Cell &cell = getCell(i, j);
      cell.setType(Cell::CellType::BOUNDARY);
      cell.setBoundary(true);
      cell.setBoundaryCondition(boundary);
    }
  }
}

// Get cells for a standard boundary location
std::vector<Cell *> Mesh::getBoundaryCells(BoundaryLocation location) {
  std::vector<Cell *> cells;

  switch (location) {
  case BoundaryLocation::LEFT:
    for (size_t j = 0; j < m_ny; ++j) {
      cells.push_back(&getCell(0, j));
    }
    break;

  case BoundaryLocation::RIGHT:
    for (size_t j = 0; j < m_ny; ++j) {
      cells.push_back(&getCell(m_nx - 1, j));
    }
    break;

  case BoundaryLocation::BOTTOM:
    for (size_t i = 0; i < m_nx; ++i) {
      cells.push_back(&getCell(i, 0));
    }
    break;

  case BoundaryLocation::TOP:
    for (size_t i = 0; i < m_nx; ++i) {
      cells.push_back(&getCell(i, m_ny - 1));
    }
    break;

  case BoundaryLocation::ALL:
    // Left and right
    for (size_t j = 0; j < m_ny; ++j) {
      cells.push_back(&getCell(0, j));
      cells.push_back(&getCell(m_nx - 1, j));
    }
    // Top and bottom (excluding corners already added)
    for (size_t i = 1; i < m_nx - 1; ++i) {
      cells.push_back(&getCell(i, 0));
      cells.push_back(&getCell(i, m_ny - 1));
    }
    break;
  }

  return cells;
}

// Apply all boundary conditions
void Mesh::applyBoundaryConditions(double dt) {
  // Loop through all cells
  for (size_t j = 0; j < m_ny; ++j) {
    for (size_t i = 0; i < m_nx; ++i) {
      Cell &cell = getCell(i, j);

      // Skip non-boundary cells
      if (!cell.isBoundary() || !cell.hasBoundaryCondition()) {
        continue;
      }

      // Get the boundary condition
      std::shared_ptr<BoundaryClass> bc = cell.getBoundaryCondition();

      // Get neighbors for this boundary cell
      std::vector<Cell *> neighbors = getBoundaryNeighbors(i, j);

      // Apply the boundary condition
      double x = physicalX(i);
      double y = physicalY(j);
      bc->apply(cell, x, y, dt, &neighbors);
    }
  }
}

// Get neighboring cells for boundary conditions
std::vector<Cell *> Mesh::getBoundaryNeighbors(size_t i, size_t j) {
  std::vector<Cell *> neighbors;

  // Add neighbors in the four cardinal directions (if within grid)
  if (i > 0)
    neighbors.push_back(&getCell(i - 1, j));
  if (i < m_nx - 1)
    neighbors.push_back(&getCell(i + 1, j));
  if (j > 0)
    neighbors.push_back(&getCell(i, j - 1));
  if (j < m_ny - 1)
    neighbors.push_back(&getCell(i, j + 1));

  return neighbors;
}

// Helper to get neighboring cells for a specific cell (N, E, S, W order)
std::array<Cell *, 4> Mesh::getNeighbors(size_t i, size_t j) {
  std::array<Cell *, 4> neighbors = {nullptr, nullptr, nullptr, nullptr};

  // North (j+1)
  if (j < m_ny - 1)
    neighbors[0] = &getCell(i, j + 1);

  // East (i+1)
  if (i < m_nx - 1)
    neighbors[1] = &getCell(i + 1, j);

  // South (j-1)
  if (j > 0)
    neighbors[2] = &getCell(i, j - 1);

  // West (i-1)
  if (i > 0)
    neighbors[3] = &getCell(i - 1, j);

  return neighbors;
}

// Get a field of data from the grid
std::vector<double> Mesh::getField(FieldType fieldType) const {
  switch (fieldType) {
  case FieldType::VELOCITY_X: {
    std::vector<double> vx, vy;
    getVelocityField(vx, vy);
    return vx;
  }
  case FieldType::VELOCITY_Y: {
    std::vector<double> vx, vy;
    getVelocityField(vx, vy);
    return vy;
  }
  case FieldType::PRESSURE:
    return getPressureField();
  case FieldType::TEMPERATURE:
    return getTemperatureField();
  case FieldType::DENSITY:
    return getDensityField();
  case FieldType::MATERIAL_ID: {
    std::vector<double> materialField(m_nx * m_ny);
    for (size_t j = 0; j < m_ny; ++j) {
      for (size_t i = 0; i < m_nx; ++i) {
        const Cell &cell = getCell(i, j);
        materialField[index(i, j)] =
            cell.getMaterial()
                ? static_cast<double>(cell.getMaterial()->getID())
                : -1.0;
      }
    }
    return materialField;
  }
  case FieldType::BOUNDARY_TYPE: {
    std::vector<double> boundaryField(m_nx * m_ny);
    for (size_t j = 0; j < m_ny; ++j) {
      for (size_t i = 0; i < m_nx; ++i) {
        const Cell &cell = getCell(i, j);
        double value = 0.0; // Default: not a boundary

        if (cell.isBoundary()) {
          if (cell.hasBoundaryCondition()) {
            // Map different boundary types to numerical values
            const std::string type = cell.getBoundaryCondition()->getType();

            if (type == "Dirichlet")
              value = 1.0;
            else if (type == "Neumann")
              value = 2.0;
            else if (type == "Inflow")
              value = 3.0;
            else if (type == "NoSlip")
              value = 4.0;
            else if (type == "Slip")
              value = 5.0;
            else if (type == "Periodic")
              value = 6.0;
            else
              value = 0.5; // Unknown boundary type
          } else {
            value = 0.5; // Boundary without condition
          }
        }

        boundaryField[index(i, j)] = value;
      }
    }
    return boundaryField;
  }
  case FieldType::CELL_TYPE: {
    std::vector<double> cellTypeField(m_nx * m_ny);
    for (size_t j = 0; j < m_ny; ++j) {
      for (size_t i = 0; i < m_nx; ++i) {
        const Cell &cell = getCell(i, j);

        // Map cell types to numerical values
        if (cell.isObstacle()) {
          cellTypeField[index(i, j)] = 2.0; // Solid/obstacle
        } else if (cell.isBoundary()) {
          cellTypeField[index(i, j)] = 1.0; // Boundary
        } else {
          cellTypeField[index(i, j)] = 0.0; // Fluid
        }
      }
    }
    return cellTypeField;
  }
  default:
    throw std::invalid_argument("Unknown field type");
  }
}

// Get velocity field components
void Mesh::getVelocityField(std::vector<double> &vx,
                            std::vector<double> &vy) const {
  vx.resize(m_nx * m_ny);
  vy.resize(m_nx * m_ny);

  for (size_t j = 0; j < m_ny; ++j) {
    for (size_t i = 0; i < m_nx; ++i) {
      size_t idx = index(i, j);
      const Cell &cell = getCell(i, j);
      vx[idx] = cell.getVelocityU();
      vy[idx] = cell.getVelocityV();
    }
  }
}

// Get the pressure field
std::vector<double> Mesh::getPressureField() const {
  std::vector<double> pressureField(m_nx * m_ny);

  for (size_t j = 0; j < m_ny; ++j) {
    for (size_t i = 0; i < m_nx; ++i) {
      pressureField[index(i, j)] = getCell(i, j).getPressure();
    }
  }

  return pressureField;
}

// Get the temperature field
std::vector<double> Mesh::getTemperatureField() const {
  std::vector<double> temperatureField(m_nx * m_ny);

  for (size_t j = 0; j < m_ny; ++j) {
    for (size_t i = 0; i < m_nx; ++i) {
      temperatureField[index(i, j)] = getCell(i, j).getTemperature();
    }
  }

  return temperatureField;
}

// Get the density field
std::vector<double> Mesh::getDensityField() const {
  std::vector<double> densityField(m_nx * m_ny);

  for (size_t j = 0; j < m_ny; ++j) {
    for (size_t i = 0; i < m_nx; ++i) {
      densityField[index(i, j)] = getCell(i, j).getDensity();
    }
  }

  return densityField;
}

// Set a cell as an obstacle
void Mesh::setCellAsObstacle(size_t i, size_t j,
                             std::shared_ptr<Material> material) {
  if (i >= m_nx || j >= m_ny) {
    throw std::out_of_range("Cell indices out of range");
  }

  Cell &cell = getCell(i, j);
  cell.setType(Cell::CellType::SOLID);
  cell.setObstacle(true);
  cell.setFixed(true);

  // Set material if provided
  if (material) {
    cell.setMaterial(material);
  }

  // Set velocity to zero (no-slip condition)
  cell.setVelocityU(0.0);
  cell.setVelocityV(0.0);
}

// Set a cell as a boundary
void Mesh::setCellAsBoundary(size_t i, size_t j,
                             std::shared_ptr<BoundaryClass> boundary) {
  if (i >= m_nx || j >= m_ny) {
    throw std::out_of_range("Cell indices out of range");
  }

  Cell &cell = getCell(i, j);
  cell.setType(Cell::CellType::BOUNDARY);
  cell.setBoundary(true);
  cell.setFixed(true);

  // Set boundary condition if provided
  if (boundary) {
    cell.setBoundaryCondition(boundary);

    // Store in boundary map if not already there
    std::string boundaryName = boundary->getName();
    if (boundaryName.empty()) {
      static int counter = 0;
      boundaryName = "Boundary_" + std::to_string(counter++);
    }

    if (m_boundaryConditions.find(boundaryName) == m_boundaryConditions.end()) {
      m_boundaryConditions[boundaryName] = boundary;
    }
  }
}

// Set a cell as fluid
void Mesh::setCellAsFluid(size_t i, size_t j,
                          std::shared_ptr<Material> material) {
  if (i >= m_nx || j >= m_ny) {
    throw std::out_of_range("Cell indices out of range");
  }

  Cell &cell = getCell(i, j);
  cell.setType(Cell::CellType::FLUID);
  cell.setBoundary(false);
  cell.setObstacle(false);
  cell.setFixed(false);

  // Clear any boundary condition
  cell.setBoundaryCondition(nullptr);

  // Set material if provided
  if (material) {
    cell.setMaterial(material);
  }
}