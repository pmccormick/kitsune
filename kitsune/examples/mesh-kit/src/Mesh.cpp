/**
 * Implementation of Mesh class with field-based storage
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
      m_dy(height / std::max(ny - 1, size_t(1))),
      // Initialize all fields with the mesh dimensions
      m_temperature(nx, ny), m_pressure(nx, ny), m_density(nx, ny),
      m_velocityU(nx, ny), m_velocityV(nx, ny), m_cellType(nx, ny),
      m_fixedStatus(nx, ny), m_boundaryFlag(nx, ny), m_obstacleFlag(nx, ny),
      m_flags(nx, ny), m_materialIds(nx, ny), m_boundaryNames(nx, ny),
      m_properties(nx, ny), m_vertexVelocityX(nx, ny),
      m_vertexVelocityY(nx, ny) {

  // Validate inputs
  if (nx < 2 || ny < 2) {
    throw std::invalid_argument("Mesh dimensions must be at least 2x2");
  }
  if (width <= 0.0 || height <= 0.0) {
    throw std::invalid_argument("Mesh dimensions must be positive");
  }

  // Initialize all fields with default values
  for (size_t j = 0; j < ny; ++j) {
    for (size_t i = 0; i < nx; ++i) {
      // Physical properties
      m_temperature(i, j) = 293.15; // Default 20°C in Kelvin
      m_pressure(i, j) = 101325.0;  // Default atmospheric pressure in Pascal
      m_density(i, j) = 1.0;        // Default density in kg/m³
      m_velocityU(i, j) = 0.0;      // Default velocity in m/s
      m_velocityV(i, j) = 0.0;      // Default velocity in m/s

      // Cell state
      m_cellType(i, j) = Cell::CellType::FLUID; // Default to fluid cells
      m_fixedStatus(i, j) = false;              // Default to non-fixed
      m_boundaryFlag(i, j) = false;             // Default to non-boundary
      m_obstacleFlag(i, j) = false;             // Default to non-obstacle
      m_flags(i, j) = 0;                        // No flags set
      m_materialIds(i, j) = 0;                  // No material assigned
      m_boundaryNames(i, j) = ""; // No boundary condition assigned

      // Initialize all properties to 0
      for (size_t p = 0; p < static_cast<size_t>(Cell::PropertyType::COUNT);
           ++p) {
        m_properties(i, j, p) = 0.0;
      }

      // Initialize vertex velocities to 0
      for (size_t v = 0; v < 4; ++v) {
        m_vertexVelocityX(i, j, v) = 0.0;
        m_vertexVelocityY(i, j, v) = 0.0;
      }

      // Set boundary cells at the domain perimeter
      if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) {
        setCellAsBoundary(i, j);
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
  return getCachedCell(i, j);
}

const Cell &Mesh::getCell(size_t i, size_t j) const {
  if (i >= m_nx || j >= m_ny) {
    throw std::out_of_range("Cell indices out of range");
  }
  return getCachedCell(i, j);
}

// Helper to get or create a cell in the cache
Cell &Mesh::getCachedCell(size_t i, size_t j) const {
  // We'll use the 1D index for cache lookup
  size_t idx = index(i, j);

  // Ensure the cache is big enough
  if (m_cellCache.size() <= idx) {
    m_cellCache.resize(m_nx * m_ny);
  }

  // Get the cached cell
  Cell &cell = m_cellCache[idx];

  // If this is a newly created cell, initialize it with mesh and indices
  if (cell.getGrid() == nullptr) {
    // Const cast is necessary here since we're modifying the cache from a const
    // method
    Mesh *self = const_cast<Mesh *>(this);
    new (&cell) Cell(self, i, j);
  }

  return cell;
}

// Material and boundary condition access
std::shared_ptr<Material> Mesh::getCellMaterial(size_t i, size_t j) const {
  uint32_t materialId = m_materialIds(i, j);
  if (materialId == 0) {
    return nullptr;
  }

  auto it = m_materials.find(materialId);
  return (it != m_materials.end()) ? it->second : nullptr;
}

void Mesh::setCellMaterial(size_t i, size_t j,
                           std::shared_ptr<Material> material) {
  if (material) {
    uint32_t materialId = registerMaterial(material);
    m_materialIds(i, j) = materialId;
  } else {
    m_materialIds(i, j) = 0;
  }
}

std::shared_ptr<BoundaryClass> Mesh::getCellBoundaryCondition(size_t i,
                                                              size_t j) const {
  std::string bcName = m_boundaryNames(i, j);
  if (bcName.empty()) {
    return nullptr;
  }

  auto it = m_boundaryConditions.find(bcName);
  return (it != m_boundaryConditions.end()) ? it->second : nullptr;
}

void Mesh::setCellBoundaryCondition(size_t i, size_t j,
                                    std::shared_ptr<BoundaryClass> bc) {
  if (bc) {
    std::string bcName = bc->getName();
    if (bcName.empty()) {
      // Generate a unique name if not provided
      static int counter = 0;
      bcName = "Boundary_" + std::to_string(counter++);
      // TODO: Modify the boundary name directly if possible
    }

    // Store in boundary map if not already there
    if (m_boundaryConditions.find(bcName) == m_boundaryConditions.end()) {
      m_boundaryConditions[bcName] = bc;
    }

    // Set the boundary name in the field
    m_boundaryNames(i, j) = bcName;
  } else {
    m_boundaryNames(i, j) = "";
  }
}

// Helper to register a material in our registry
uint32_t Mesh::registerMaterial(std::shared_ptr<Material> material) {
  // If the material already has a valid ID in our registry, use it
  uint32_t materialId = material->getID();
  auto it = m_materials.find(materialId);
  if (it != m_materials.end() && it->second == material) {
    return materialId;
  }

  // Otherwise, assign a new ID
  uint32_t newId = m_nextMaterialId++;
  m_materials[newId] = material;
  return newId;
}

// Initialize the mesh with a default material
void Mesh::initialize(std::shared_ptr<Material> defaultMaterial) {
  if (!defaultMaterial) {
    throw std::invalid_argument("Default material cannot be null");
  }

  uint32_t materialId = registerMaterial(defaultMaterial);

  // Set the material for all cells
  for (size_t j = 0; j < m_ny; ++j) {
    for (size_t i = 0; i < m_nx; ++i) {
      m_materialIds(i, j) = materialId;
    }
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
      setCellAsBoundary(i, j, boundary);
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
      // Check if this is a boundary cell with a boundary condition
      if (m_boundaryFlag(i, j) && !m_boundaryNames(i, j).empty()) {
        // Get the cell and its boundary condition
        Cell &cell = getCell(i, j);
        std::shared_ptr<BoundaryClass> bc = cell.getBoundaryCondition();

        if (bc) {
          // Get neighbors for this boundary cell
          std::vector<Cell *> neighbors = getBoundaryNeighbors(i, j);

          // Apply the boundary condition
          double x = physicalX(i);
          double y = physicalY(j);
          bc->apply(cell, x, y, dt, &neighbors);
        }
      }
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
// This snippet shows the changes needed in Mesh.cpp to support the new
// enum-based approach Specifically focusing on the getField method where
// boundary type checks are performed

std::vector<double> Mesh::getField(FieldType fieldType) const {
  std::vector<double> result(m_nx * m_ny);

  for (size_t j = 0; j < m_ny; ++j) {
    for (size_t i = 0; i < m_nx; ++i) {
      size_t idx = index(i, j);

      switch (fieldType) {
      case FieldType::VELOCITY_X:
        result[idx] = m_velocityU(i, j);
        break;
      case FieldType::VELOCITY_Y:
        result[idx] = m_velocityV(i, j);
        break;
      case FieldType::PRESSURE:
        result[idx] = m_pressure(i, j);
        break;
      case FieldType::TEMPERATURE:
        result[idx] = m_temperature(i, j);
        break;
      case FieldType::DENSITY:
        result[idx] = m_density(i, j);
        break;
      case FieldType::MATERIAL_ID:
        result[idx] = static_cast<double>(m_materialIds(i, j));
        break;
      case FieldType::BOUNDARY_TYPE: {
        double value = 0.0; // Default: not a boundary
        if (m_boundaryFlag(i, j)) {
          if (!m_boundaryNames(i, j).empty()) {
            // Map different boundary types to numerical values using the enum
            auto it = m_boundaryConditions.find(m_boundaryNames(i, j));
            if (it != m_boundaryConditions.end()) {
              // Use the enum directly for faster, type-safe mapping
              BoundaryType type = it->second->getTypeEnum();
              // Map each enum value to a unique numeric value for visualization
              switch (type) {
              case BoundaryType::DIRICHLET:
                value = 1.0;
                break;
              case BoundaryType::NEUMANN:
                value = 2.0;
                break;
              case BoundaryType::INFLOW:
                value = 3.0;
                break;
              case BoundaryType::NO_SLIP:
                value = 4.0;
                break;
              case BoundaryType::SLIP:
                value = 5.0;
                break;
              case BoundaryType::PERIODIC:
                value = 6.0;
                break;
              case BoundaryType::OUTFLOW:
                value = 7.0;
                break;
              case BoundaryType::SYMMETRY:
                value = 8.0;
                break;
              case BoundaryType::UNKNOWN:
              default:
                value = 0.5; // Unknown boundary type
                break;
              }
            }
          } else {
            value = 0.5; // Boundary without condition
          }
        }
        result[idx] = value;
        break;
      }
      case FieldType::CELL_TYPE: {
        double value = 0.0; // Default: fluid
        if (m_obstacleFlag(i, j)) {
          value = 2.0; // Solid/obstacle
        } else if (m_boundaryFlag(i, j)) {
          value = 1.0; // Boundary
        }
        result[idx] = value;
        break;
      }
      default:
        result[idx] = 0.0;
        break;
      }
    }
  }

  return result;
}


// Get velocity field components
void Mesh::getVelocityField(std::vector<double> &vx,
                            std::vector<double> &vy) const {
  vx.resize(m_nx * m_ny);
  vy.resize(m_nx * m_ny);

  for (size_t j = 0; j < m_ny; ++j) {
    for (size_t i = 0; i < m_nx; ++i) {
      size_t idx = index(i, j);
      vx[idx] = m_velocityU(i, j);
      vy[idx] = m_velocityV(i, j);
    }
  }
}

// Get the pressure field
std::vector<double> Mesh::getPressureField() const {
  std::vector<double> result(m_nx * m_ny);

  for (size_t j = 0; j < m_ny; ++j) {
    for (size_t i = 0; i < m_nx; ++i) {
      result[index(i, j)] = m_pressure(i, j);
    }
  }

  return result;
}

// Get the temperature field
std::vector<double> Mesh::getTemperatureField() const {
  std::vector<double> result(m_nx * m_ny);

  for (size_t j = 0; j < m_ny; ++j) {
    for (size_t i = 0; i < m_nx; ++i) {
      result[index(i, j)] = m_temperature(i, j);
    }
  }

  return result;
}

// Get the density field
std::vector<double> Mesh::getDensityField() const {
  std::vector<double> result(m_nx * m_ny);

  for (size_t j = 0; j < m_ny; ++j) {
    for (size_t i = 0; i < m_nx; ++i) {
      result[index(i, j)] = m_density(i, j);
    }
  }

  return result;
}

// Set a cell as an obstacle
void Mesh::setCellAsObstacle(size_t i, size_t j,
                             std::shared_ptr<Material> material) {
  if (i >= m_nx || j >= m_ny) {
    throw std::out_of_range("Cell indices out of range");
  }

  // Update the cell type and flags
  m_cellType(i, j) = Cell::CellType::SOLID;
  m_obstacleFlag(i, j) = true;
  m_boundaryFlag(i, j) = false;
  m_fixedStatus(i, j) = true;

  // Set material if provided
  if (material) {
    setCellMaterial(i, j, material);
  }

  // Set velocity to zero (no-slip condition)
  m_velocityU(i, j) = 0.0;
  m_velocityV(i, j) = 0.0;
}

// Set a cell as a boundary
void Mesh::setCellAsBoundary(size_t i, size_t j,
                             std::shared_ptr<BoundaryClass> boundary) {
  if (i >= m_nx || j >= m_ny) {
    throw std::out_of_range("Cell indices out of range");
  }

  // Update the cell type and flags
  m_cellType(i, j) = Cell::CellType::BOUNDARY;
  m_boundaryFlag(i, j) = true;
  m_obstacleFlag(i, j) = false;
  m_fixedStatus(i, j) = true;

  // Set boundary condition if provided
  if (boundary) {
    setCellBoundaryCondition(i, j, boundary);
  }
}

// Set a cell as fluid
void Mesh::setCellAsFluid(size_t i, size_t j,
                          std::shared_ptr<Material> material) {
  if (i >= m_nx || j >= m_ny) {
    throw std::out_of_range("Cell indices out of range");
  }

  // Update the cell type and flags
  m_cellType(i, j) = Cell::CellType::FLUID;
  m_boundaryFlag(i, j) = false;
  m_obstacleFlag(i, j) = false;
  m_fixedStatus(i, j) = false;

  // Clear any boundary condition
  m_boundaryNames(i, j) = "";

  // Set material if provided
  if (material) {
    setCellMaterial(i, j, material);
  }
}