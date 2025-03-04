#include "Grid.h"
#include <algorithm>
#include <cmath>

Grid::Grid(size_t nx, size_t ny, double width, double height, double origin_x,
           double origin_y)
    : m_nx(nx), m_ny(ny), m_width(width), m_height(height),
      m_origin_x(origin_x), m_origin_y(origin_y) {

  if (nx < 2 || ny < 2) {
    throw std::invalid_argument("Grid dimensions must be at least 2x2");
  }

  m_dx = m_width / (m_nx - 1);
  m_dy = m_height / (m_ny - 1);

  // Pre-allocate cells for performance
  m_cells.resize(m_nx * m_ny);

  // Mark boundary cells
  for (size_t i = 0; i < m_nx; ++i) {
    getCell(i, 0).setBoundary(true);
    getCell(i, m_ny - 1).setBoundary(true);
  }

  for (size_t j = 0; j < m_ny; ++j) {
    getCell(0, j).setBoundary(true);
    getCell(m_nx - 1, j).setBoundary(true);
  }
}

Cell *getCellData() const { return m_grid ? m_grid->getCellData() : nullptr; }

// Get velocity field for visualization or analysis
void Grid::getVelocityField(std::vector<double> &vx,
                            std::vector<double> &vy) const {
  vx.resize(m_nx * m_ny);
  vy.resize(m_nx * m_ny);

  for (size_t j = 0; j < m_ny; ++j) {
    for (size_t i = 0; i < m_nx; ++i) {
      size_t idx = index(i, j);
      const Cell &cell = m_cells[idx];
      vx[idx] = cell.getVelocityX();
      vy[idx] = cell.getVelocityY();
    }
  }
}

// Get pressure field for visualization or analysis
std::vector<double> Grid::getPressureField() const {
  std::vector<double> pressureField(m_nx * m_ny);

  for (size_t j = 0; j < m_ny; ++j) {
    for (size_t i = 0; i < m_nx; ++i) {
      pressureField[index(i, j)] = getCell(i, j).getPressure();
    }
  }

  return pressureField;
}

std::vector<double> Grid::getTemperatureField() const {
  std::vector<double> temperatureField(m_nx * m_ny);
  size_t idx = 0;

  for (size_t j = 0; j < m_ny; ++j) {
    for (size_t i = 0; i < m_nx; ++i) {
      temperatureField[index(i, j)] = getCell(i, j).getTemperature();
    }
  }

  return temperatureField;
}

// Calculate divergence at a cell (useful for pressure solvers)
double Grid::calculateDivergence(size_t i, size_t j) const {
  if (i == 0 || i == m_nx - 1 || j == 0 || j == m_ny - 1) {
    return 0.0; // Zero divergence at boundaries
  }

  const double vx_right = getCell(i + 1, j).getVelocityX();
  const double vx_left = getCell(i - 1, j).getVelocityX();
  const double vy_top = getCell(i, j + 1).getVelocityY();
  const double vy_bottom = getCell(i, j - 1).getVelocityY();

  return (vx_right - vx_left) / (2.0 * m_dx) +
         (vy_top - vy_bottom) / (2.0 * m_dy);
}

// Define a circular obstacle
void Grid::setCircularObstacle(double center_x, double center_y, double radius,
                               Material *material) {
  if (!material) {
    throw std::invalid_argument("Material cannot be null");
  }

  // Convert physical coordinates to grid indices
  size_t center_i = gridI(center_x);
  size_t center_j = gridJ(center_y);
  size_t radius_cells =
      static_cast<size_t>(radius / std::min(m_dx, m_dy) + 0.5);

  // Set cells within radius as obstacles with the specified material
  for (size_t j = std::max(center_j, radius_cells) - radius_cells;
       j <= std::min(center_j + radius_cells, m_ny - 1); ++j) {
    for (size_t i = std::max(center_i, radius_cells) - radius_cells;
         i <= std::min(center_i + radius_cells, m_nx - 1); ++i) {

      double dist_sq = std::pow((static_cast<double>(i) - center_i) * m_dx, 2) +
                       std::pow((static_cast<double>(j) - center_j) * m_dy, 2);

      if (dist_sq <= radius * radius) {
        Cell &cell = getCell(i, j);
        cell.setObstacle(true);
        cell.setMaterial(material);
        cell.setVelocityX(0.0); // No-slip condition for obstacles
        cell.setVelocityY(0.0);
      }
    }
  }
}

void Grid::initializeTemperature(double defaultTemp) {
  // Initialize temperature in each cell
  for (size_t j = 0; j < m_ny; ++j) {
    for (size_t i = 0; i < m_nx; ++i) {
      getCell(i, j).setTemperature(defaultTemp);
    }
  }
}

void Grid::setTemperature(size_t i, size_t j, double temp) {
  if (i < m_nx && j < m_ny) {
    getCell(i, j).setTemperature(temp);
  } else {
    throw std::out_of_range("Grid::setTemperature: Indices out of range");
  }
}

double Grid::getTemperature(size_t i, size_t j) const {
  if (i < m_nx && j < m_ny) {
    return getCell(i, j).getTemperature();
  } else {
    throw std::out_of_range("Grid::getTemperature: Indices out of range");
  }
}

double Grid::interpolateTemperature(double x, double y) const {
  // Get grid cell indices for the position
  int i = static_cast<int>((x - m_origin_x) / m_dx);
  int j = static_cast<int>((y - m_origin_y) / m_dy);

  // Check bounds
  if (i < 0 || i >= m_nx - 1 || j < 0 || j >= m_ny - 1) {
    throw std::out_of_range("Position outside grid bounds");
  }

  // Calculate fractional position within cell
  double x_frac = (x - (m_origin_x + i * m_dx)) / m_dx;
  double y_frac = (y - (m_origin_y + j * m_dy)) / m_dy;

  // Get values at cell corners
  double t00 = getCell(i, j).getTemperature();
  double t10 = getCell(i + 1, j).getTemperature();
  double t01 = getCell(i, j + 1).getTemperature();
  double t11 = getCell(i + 1, j + 1).getTemperature();

  // Bilinear interpolation
  double tx0 = t00 * (1.0 - x_frac) + t10 * x_frac;
  double tx1 = t01 * (1.0 - x_frac) + t11 * x_frac;

  return tx0 * (1.0 - y_frac) + tx1 * y_frac;
}

static std::shared_ptr<Grid>
Grid::createWithUnits(size_t nx, size_t ny, double width, double height,
                      const std::string &lengthUnit, double originX = 0.0,
                      double originY = 0.0) {

  // Convert all dimensions to SI (meters)
  double widthMeters = Units::convert(width, lengthUnit, "m");
  double heightMeters = Units::convert(height, lengthUnit, "m");
  double originXMeters = Units::convert(originX, lengthUnit, "m");
  double originYMeters = Units::convert(originY, lengthUnit, "m");

  return std::make_shared<Grid>(nx, ny, widthMeters, heightMeters,
                                originXMeters, originYMeters);
}

/**
 * @brief Initialize density of all cells with unit conversion
 * @param defaultDensity Default density in specified units
 * @param densityUnit Density unit string (e.g., "kg/m³", "lb/ft³")
 */
void Grid::initializeDensityWithUnits(
    double defaultDensity, const std::string &densityUnit = "kg/m³") {
  double densityKgM3 = Units::convert(defaultDensity, densityUnit, "kg/m³");

  for (size_t j = 0; j < m_ny; ++j) {
    for (size_t i = 0; i < m_nx; ++i) {
      getCell(i, j).setDensity(densityKgM3);
    }
  }
}

/**
 * @brief Set density of a specific cell with unit conversion
 * @param i Grid x-index
 * @param j Grid y-index
 * @param density Density value in specified units
 * @param densityUnit Density unit string (e.g., "kg/m³", "lb/ft³")
 */
void Grid::setDensityWithUnits(size_t i, size_t j, double density,
                               const std::string &densityUnit = "kg/m³") {
  double densityKgM3 = Units::convert(density, densityUnit, "kg/m³");
  getCell(i, j).setDensity(densityKgM3);
}

/**
 * @brief Set density for a region of cells with unit conversion
 * @param i_start Starting x-index
 * @param i_end Ending x-index (inclusive)
 * @param j_start Starting y-index
 * @param j_end Ending y-index (inclusive)
 * @param density Density value in specified units
 * @param densityUnit Density unit string (e.g., "kg/m³", "lb/ft³")
 */
void Grid::setRegionDensityWithUnits(size_t i_start, size_t i_end,
                                     size_t j_start, size_t j_end,
                                     double density,
                                     const std::string &densityUnit = "kg/m³") {
  double densityKgM3 = Units::convert(density, densityUnit, "kg/m³");

  for (size_t j = j_start; j <= j_end && j < m_ny; ++j) {
    for (size_t i = i_start; i <= i_end && i < m_nx; ++i) {
      getCell(i, j).setDensity(densityKgM3);
    }
  }
}

/**
 * @brief Get density of a specific cell with unit conversion
 * @param i Grid x-index
 * @param j Grid y-index
 * @param densityUnit Density unit string (e.g., "kg/m³", "lb/ft³")
 * @return Density in specified units
 */
double
Grid::getDensityWithUnits(size_t i, size_t j,
                          const std::string &densityUnit = "kg/m³") const {
  double densityKgM3 = getCell(i, j).getDensity();
  return Units::convert(densityKgM3, "kg/m³", densityUnit);
}

/**
 * @brief Get density field with unit conversion
 * @param densityUnit Density unit string (e.g., "kg/m³", "lb/ft³")
 * @return Vector of density values in specified units
 */
std::vector<double>
Grid::getDensityFieldWithUnits(const std::string &densityUnit = "kg/m³") const {

  std::vector<double> densityField(m_nx * m_ny);

  for (size_t j = 0; j < m_ny; ++j) {
    for (size_t i = 0; i < m_nx; ++i) {
      size_t idx = index(i, j);
      double densityKgM3 = getCell(i, j).getDensity();
      densityField[idx] = Units::convert(densityKgM3, "kg/m³", densityUnit);
    }
  }

  return densityField;
}

/**
 * @brief Apply a density stratification with unit conversion (e.g., for
 * simulating atmospheric or ocean layers)
 * @param baseValue Base density value in specified units
 * @param gradient Density change per unit distance in baseUnit/lengthUnit
 * @param direction Direction of stratification ('x', 'y')
 * @param densityUnit Density unit string (e.g., "kg/m³", "lb/ft³")
 * @param lengthUnit Length unit for gradient (e.g., "m", "ft")
 */
void Grid::applyDensityStratificationWithUnits(
    double baseValue, double gradient, char direction = 'y',
    const std::string &densityUnit = "kg/m³",
    const std::string &lengthUnit = "m") {
  double baseValueKgM3 = Units::convert(baseValue, densityUnit, "kg/m³");

  // Convert gradient to SI units (kg/m³ per meter)
  double gradientPerM = gradient;
  if (densityUnit != "kg/m³" || lengthUnit != "m") {
    // First convert to a change in the original density unit per meter
    double changePerMeter = gradient * Units::convert(1.0, lengthUnit, "m");
    // Then convert that density change to kg/m³
    gradientPerM = Units::convert(changePerMeter, densityUnit, "kg/m³");
  }

  for (size_t j = 0; j < m_ny; ++j) {
    for (size_t i = 0; i < m_nx; ++i) {
      double distanceM = 0.0;

      if (direction == 'y') {
        distanceM = physicalY(j) - m_origin_y;
      } else { // 'x' direction
        distanceM = physicalX(i) - m_origin_x;
      }

      double density = baseValueKgM3 + gradientPerM * distanceM;
      // Ensure density is physically valid
      density = Units::enforceValidDensity(density);

      getCell(i, j).setDensity(density);
    }
  }
}

/**
 * @brief Initialize pressure of all cells with unit conversion
 * @param defaultPressure Default pressure in specified units
 * @param pressureUnit Pressure unit string (e.g., "Pa", "bar", "atm")
 */
void Grid::initializePressureWithUnits(double defaultPressure,
                                       const std::string &pressureUnit = "Pa") {
  double pressurePa = Units::convert(defaultPressure, pressureUnit, "Pa");

  for (size_t j = 0; j < m_ny; ++j) {
    for (size_t i = 0; i < m_nx; ++i) {
      getCell(i, j).setPressure(pressurePa);
    }
  }
}

/**
 * @brief Set pressure of a specific cell with unit conversion
 * @param i Grid x-index
 * @param j Grid y-index
 * @param pressure Pressure value in specified units
 * @param pressureUnit Pressure unit string (e.g., "Pa", "bar", "atm")
 */
void Grid::setPressureWithUnits(size_t i, size_t j, double pressure,
                                const std::string &pressureUnit = "Pa") {
  double pressurePa = Units::convert(pressure, pressureUnit, "Pa");
  getCell(i, j).setPressure(pressurePa);
}

/**
 * @brief Set pressure for a region of cells with unit conversion
 * @param i_start Starting x-index
 * @param i_end Ending x-index (inclusive)
 * @param j_start Starting y-index
 * @param j_end Ending y-index (inclusive)
 * @param pressure Pressure value in specified units
 * @param pressureUnit Pressure unit string (e.g., "Pa", "bar", "atm")
 */
void Grid::setRegionPressureWithUnits(size_t i_start, size_t i_end,
                                      size_t j_start, size_t j_end,
                                      double pressure,
                                      const std::string &pressureUnit = "Pa") {
  double pressurePa = Units::convert(pressure, pressureUnit, "Pa");

  for (size_t j = j_start; j <= j_end && j < m_ny; ++j) {
    for (size_t i = i_start; i <= i_end && i < m_nx; ++i) {
      getCell(i, j).setPressure(pressurePa);
    }
  }
}

/**
 * @brief Get pressure of a specific cell with unit conversion
 * @param i Grid x-index
 * @param j Grid y-index
 * @param pressureUnit Pressure unit string (e.g., "Pa", "bar", "atm")
 * @return Pressure in specified units
 */
double
Grid::getPressureWithUnits(size_t i, size_t j,
                           const std::string &pressureUnit = "Pa") const {
  double pressurePa = getCell(i, j).getPressure();
  return Units::convert(pressurePa, "Pa", pressureUnit);
}

/**
 * @brief Get entire pressure field with unit conversion
 * @param pressureUnit Pressure unit string (e.g., "Pa", "bar", "atm")
 * @return Vector of pressure values in specified units
 */
std::vector<double>
Grid::getPressureFieldWithUnits(const std::string &pressureUnit = "Pa") const {
  std::vector<double> fieldPa = getPressureField();

  if (pressureUnit == "Pa") {
    return fieldPa; // No conversion needed
  }

  std::vector<double> fieldConverted(fieldPa.size());
  for (size_t i = 0; i < fieldPa.size(); ++i) {
    fieldConverted[i] = Units::convert(fieldPa[i], "Pa", pressureUnit);
  }

  return fieldConverted;
}

/**
 * @brief Apply a pressure gradient with unit conversion
 * @param startPressure Starting pressure in specified units
 * @param endPressure Ending pressure in specified units
 * @param direction Direction of gradient ('x', 'y', or 'radial')
 * @param pressureUnit Pressure unit string (e.g., "Pa", "bar", "atm")
 */
void Grid::applyPressureGradientWithUnits(
    double startPressure, double endPressure, char direction = 'x',
    const std::string &pressureUnit = "Pa") {
  double startPressurePa = Units::convert(startPressure, pressureUnit, "Pa");
  double endPressurePa = Units::convert(endPressure, pressureUnit, "Pa");

  // Simple linear gradient implementation
  for (size_t j = 0; j < m_ny; ++j) {
    for (size_t i = 0; i < m_nx; ++i) {
      double fraction = 0.0;

      switch (direction) {
      case 'x':
        fraction = static_cast<double>(i) / (m_nx - 1);
        break;
      case 'y':
        fraction = static_cast<double>(j) / (m_ny - 1);
        break;
      case 'r': { // Radial from center
        double centerX = m_nx / 2.0;
        double centerY = m_ny / 2.0;
        double maxDist = std::sqrt(centerX * centerX + centerY * centerY);
        double dist =
            std::sqrt(std::pow(i - centerX, 2) + std::pow(j - centerY, 2));
        fraction = dist / maxDist;
        break;
      }
      default:
        fraction = static_cast<double>(i) / (m_nx - 1);
      }

      double pressure =
          startPressurePa + fraction * (endPressurePa - startPressurePa);
      getCell(i, j).setPressure(pressure);
    }
  }
}

/**
 * @brief Initialize temperature of all cells with unit conversion
 * @param defaultTemp Default temperature in specified units
 * @param tempUnit Temperature unit string (e.g., "K", "C", "F")
 */
void Grid::initializeTemperatureWithUnits(double defaultTemp,
                                          const std::string &tempUnit = "K") {
  double tempKelvin = Units::convert(defaultTemp, tempUnit, "K");
  initializeTemperature(tempKelvin);
}

/**
 * @brief Set temperature of a specific cell with unit conversion
 * @param i Grid x-index
 * @param j Grid y-index
 * @param temp Temperature value in specified units
 * @param tempUnit Temperature unit string (e.g., "K", "C", "F")
 */
void Grid::setTemperatureWithUnits(size_t i, size_t j, double temp,
                                   const std::string &tempUnit = "K") {
  double tempKelvin = Units::convert(temp, tempUnit, "K");
  setTemperature(i, j, tempKelvin);
}

/**
 * @brief Set temperature for a region of cells with unit conversion
 * @param i_start Starting x-index
 * @param i_end Ending x-index (inclusive)
 * @param j_start Starting y-index
 * @param j_end Ending y-index (inclusive)
 * @param temp Temperature value in specified units
 * @param tempUnit Temperature unit string (e.g., "K", "C", "F")
 */
void Grid::setRegionTemperatureWithUnits(size_t i_start, size_t i_end,
                                         size_t j_start, size_t j_end,
                                         double temp,
                                         const std::string &tempUnit = "K") {
  double tempKelvin = Units::convert(temp, tempUnit, "K");

  for (size_t j = j_start; j <= j_end && j < m_ny; ++j) {
    for (size_t i = i_start; i <= i_end && i < m_nx; ++i) {
      setTemperature(i, j, tempKelvin);
    }
  }
}

/**
 * @brief Get temperature of a specific cell with unit conversion
 * @param i Grid x-index
 * @param j Grid y-index
 * @param tempUnit Temperature unit string (e.g., "K", "C", "F")
 * @return Temperature in specified units
 */
double Grid::getTemperatureWithUnits(size_t i, size_t j,
                                     const std::string &tempUnit = "K") const {
  double tempKelvin = getTemperature(i, j);
  return Units::convert(tempKelvin, "K", tempUnit);
}

/**
 * @brief Get entire temperature field with unit conversion
 * @param tempUnit Temperature unit string (e.g., "K", "C", "F")
 * @return Vector of temperature values in specified units
 */
std::vector<double>
Grid::getTemperatureFieldWithUnits(const std::string &tempUnit = "K") const {
  std::vector<double> fieldKelvin = getTemperatureField();

  if (tempUnit == "K") {
    return fieldKelvin; // No conversion needed
  }

  std::vector<double> fieldConverted(fieldKelvin.size());
  for (size_t i = 0; i < fieldKelvin.size(); ++i) {
    fieldConverted[i] = Units::convert(fieldKelvin[i], "K", tempUnit);
  }

  return fieldConverted;
}

/**
 * @brief Interpolate temperature at a physical position with unit conversion
 * @param x Physical x-coordinate in specified length units
 * @param y Physical y-coordinate in specified length units
 * @param lengthUnit Length unit string (e.g., "m", "ft", "in")
 * @param tempUnit Temperature unit string (e.g., "K", "C", "F")
 * @return Interpolated temperature in specified units
 */
double
Grid::interpolateTemperatureWithUnits(double x, double y,
                                      const std::string &lengthUnit = "m",
                                      const std::string &tempUnit = "K") const {
  double xMeters = Units::convert(x, lengthUnit, "m");
  double yMeters = Units::convert(y, lengthUnit, "m");

  double tempKelvin = interpolateTemperature(xMeters, yMeters);
  return Units::convert(tempKelvin, "K", tempUnit);
}

/**
 * @brief Apply a temperature gradient with unit conversion
 * @param startTemp Starting temperature in specified units
 * @param endTemp Ending temperature in specified units
 * @param direction Direction of gradient ('x', 'y', or 'radial')
 * @param tempUnit Temperature unit string (e.g., "K", "C", "F")
 */
void Grid::applyTemperatureGradientWithUnits(
    double startTemp, double endTemp, char direction = 'x',
    const std::string &tempUnit = "K") {
  double startTempK = Units::convert(startTemp, tempUnit, "K");
  double endTempK = Units::convert(endTemp, tempUnit, "K");

  // Simple linear gradient implementation
  for (size_t j = 0; j < m_ny; ++j) {
    for (size_t i = 0; i < m_nx; ++i) {
      double fraction = 0.0;

      switch (direction) {
      case 'x':
        fraction = static_cast<double>(i) / (m_nx - 1);
        break;
      case 'y':
        fraction = static_cast<double>(j) / (m_ny - 1);
        break;
      case 'r': { // Radial from center
        double centerX = m_nx / 2.0;
        double centerY = m_ny / 2.0;
        double maxDist = std::sqrt(centerX * centerX + centerY * centerY);
        double dist =
            std::sqrt(std::pow(i - centerX, 2) + std::pow(j - centerY, 2));
        fraction = dist / maxDist;
        break;
      }
      default:
        fraction = static_cast<double>(i) / (m_nx - 1);
      }

      double temp = startTempK + fraction * (endTempK - startTempK);
      setTemperature(i, j, temp);
    }
  }
}

/**
 * @brief Initialize velocity field to zero or a specified value with unit
 * conversion
 * @param defaultVx Default x-velocity in specified units
 * @param defaultVy Default y-velocity in specified units
 * @param velocityUnit Velocity unit string (e.g., "m/s", "mph", "knot")
 */
void Grid::initializeVelocityFieldWithUnits(
    double defaultVx = 0.0, double defaultVy = 0.0,
    const std::string &velocityUnit = "m/s") {
  double vxMps = Units::convert(defaultVx, velocityUnit, "m/s");
  double vyMps = Units::convert(defaultVy, velocityUnit, "m/s");

  for (size_t j = 0; j < m_ny; ++j) {
    for (size_t i = 0; i < m_nx; ++i) {
      Cell &cell = getCell(i, j);
      if (!cell.isObstacle()) { // Don't set velocity for obstacle cells
        cell.setVelocityX(vxMps);
        cell.setVelocityY(vyMps);
      }
    }
  }
}

/**
 * @brief Set velocity of a specific cell with unit conversion
 * @param i Grid x-index
 * @param j Grid y-index
 * @param vx X-velocity component in specified units
 * @param vy Y-velocity component in specified units
 * @param velocityUnit Velocity unit string (e.g., "m/s", "mph", "knot")
 */
void Grid::setVelocityWithUnits(size_t i, size_t j, double vx, double vy,
                                const std::string &velocityUnit = "m/s") {
  double vxMps = Units::convert(vx, velocityUnit, "m/s");
  double vyMps = Units::convert(vy, velocityUnit, "m/s");

  Cell &cell = getCell(i, j);
  if (!cell.isObstacle()) {
    cell.setVelocityX(vxMps);
    cell.setVelocityY(vyMps);
  }
}

/**
 * @brief Get velocity of a specific cell with unit conversion
 * @param i Grid x-index
 * @param j Grid y-index
 * @param velocityUnit Velocity unit string (e.g., "m/s", "mph", "knot")
 * @return Pair of (vx, vy) velocity components in specified units
 */
std::pair<double, double>
Grid::getVelocityWithUnits(size_t i, size_t j,
                           const std::string &velocityUnit = "m/s") const {
  const Cell &cell = getCell(i, j);
  double vxMps = cell.getVelocityX();
  double vyMps = cell.getVelocityY();

  return {Units::convert(vxMps, "m/s", velocityUnit),
          Units::convert(vyMps, "m/s", velocityUnit)};
}

/**
 * @brief Get velocity field with unit conversion
 * @param vx Output vector for x-velocity components
 * @param vy Output vector for y-velocity components
 * @param velocityUnit Velocity unit string (e.g., "m/s", "mph", "knot")
 */
void Grid::getVelocityFieldWithUnits(
    std::vector<double> &vx, std::vector<double> &vy,
    const std::string &velocityUnit = "m/s") const {
  // First get the velocity field in m/s
  getVelocityField(vx, vy);

  // Skip conversion if already in m/s
  if (velocityUnit == "m/s") {
    return;
  }

  // Convert to requested units
  for (size_t i = 0; i < vx.size(); ++i) {
    vx[i] = Units::convert(vx[i], "m/s", velocityUnit);
    vy[i] = Units::convert(vy[i], "m/s", velocityUnit);
  }
}

/**
 * @brief Apply a uniform flow in a specific direction with unit conversion
 * @param speed Flow speed magnitude in specified units
 * @param angle Flow direction angle in degrees (0=right, 90=up, etc.)
 * @param velocityUnit Velocity unit string (e.g., "m/s", "mph", "knot")
 */
void Grid::applyUniformFlowWithUnits(double speed, double angle = 0.0,
                                     const std::string &velocityUnit = "m/s") {
  double speedMps = Units::convert(speed, velocityUnit, "m/s");

  // Convert angle to radians
  double angleRad = angle * M_PI / 180.0;

  // Calculate velocity components
  double vxMps = speedMps * std::cos(angleRad);
  double vyMps = speedMps * std::sin(angleRad);

  for (size_t j = 0; j < m_ny; ++j) {
    for (size_t i = 0; i < m_nx; ++i) {
      Cell &cell = getCell(i, j);
      if (!cell.isObstacle() && !cell.isBoundary()) {
        cell.setVelocityX(vxMps);
        cell.setVelocityY(vyMps);
      }
    }
  }
}

/**
 * @brief Apply a shear flow (velocity varying in perpendicular direction) with
 * unit conversion
 * @param minSpeed Minimum flow speed in specified units
 * @param maxSpeed Maximum flow speed in specified units
 * @param direction Primary flow direction ('x' or 'y')
 * @param velocityUnit Velocity unit string (e.g., "m/s", "mph", "knot")
 */
void Grid::applyShearFlowWithUnits(double minSpeed, double maxSpeed,
                                   char direction = 'x',
                                   const std::string &velocityUnit = "m/s") {
  double minSpeedMps = Units::convert(minSpeed, velocityUnit, "m/s");
  double maxSpeedMps = Units::convert(maxSpeed, velocityUnit, "m/s");

  for (size_t j = 0; j < m_ny; ++j) {
    for (size_t i = 0; i < m_nx; ++i) {
      Cell &cell = getCell(i, j);
      if (cell.isObstacle() || cell.isBoundary()) {
        continue;
      }

      double fraction = 0.0;
      if (direction == 'x') {
        // Flow in x-direction, varying with y
        fraction = static_cast<double>(j) / (m_ny - 1);
        double vx = minSpeedMps + fraction * (maxSpeedMps - minSpeedMps);
        cell.setVelocityX(vx);
        cell.setVelocityY(0.0);
      } else {
        // Flow in y-direction, varying with x
        fraction = static_cast<double>(i) / (m_nx - 1);
        double vy = minSpeedMps + fraction * (maxSpeedMps - minSpeedMps);
        cell.setVelocityX(0.0);
        cell.setVelocityY(vy);
      }
    }
  }
}

/**
 * @brief Apply a parabolic flow profile (like in a pipe) with unit conversion
 * @param maxSpeed Maximum flow speed (at center) in specified units
 * @param direction Primary flow direction ('x' or 'y')
 * @param velocityUnit Velocity unit string (e.g., "m/s", "mph", "knot")
 */
void Grid::applyParabolicFlowWithUnits(
    double maxSpeed, char direction = 'x',
    const std::string &velocityUnit = "m/s") {
  double maxSpeedMps = Units::convert(maxSpeed, velocityUnit, "m/s");

  double centerX = m_nx / 2.0;
  double centerY = m_ny / 2.0;
  double radiusX = m_nx / 2.0;
  double radiusY = m_ny / 2.0;

  for (size_t j = 0; j < m_ny; ++j) {
    for (size_t i = 0; i < m_nx; ++i) {
      Cell &cell = getCell(i, j);
      if (cell.isObstacle() || cell.isBoundary()) {
        continue;
      }

      // Calculate normalized distance from centerline
      double normalizedDist = 0.0;
      if (direction == 'x') {
        // Flow in x-direction with parabolic profile along y
        normalizedDist = std::abs(j - centerY) / radiusY;
      } else {
        // Flow in y-direction with parabolic profile along x
        normalizedDist = std::abs(i - centerX) / radiusX;
      }

      // Parabolic profile: v = vmax * (1 - (r/R)²)
      double factor = std::max(0.0, 1.0 - normalizedDist * normalizedDist);
      double speed = maxSpeedMps * factor;

      if (direction == 'x') {
        cell.setVelocityX(speed);
        cell.setVelocityY(0.0);
      } else {
        cell.setVelocityX(0.0);
        cell.setVelocityY(speed);
      }
    }
  }
}

/**
 * @brief Apply a vortex flow (circular motion) with unit conversion
 * @param centerI Grid x-index of vortex center
 * @param centerJ Grid y-index of vortex center
 * @param maxSpeed Maximum tangential speed in specified units
 * @param radius Radius of maximum speed in grid cells
 * @param velocityUnit Velocity unit string (e.g., "m/s", "mph", "knot")
 */
void Grid::applyVortexFlowWithUnits(size_t centerI, size_t centerJ,
                                    double maxSpeed, double radius,
                                    const std::string &velocityUnit = "m/s") {
  double maxSpeedMps = Units::convert(maxSpeed, velocityUnit, "m/s");

  for (size_t j = 0; j < m_ny; ++j) {
    for (size_t i = 0; i < m_nx; ++i) {
      Cell &cell = getCell(i, j);
      if (cell.isObstacle() || cell.isBoundary()) {
        continue;
      }

      // Calculate position relative to center
      double dx = static_cast<double>(i) - centerI;
      double dy = static_cast<double>(j) - centerJ;
      double distance = std::sqrt(dx * dx + dy * dy);

      // Calculate tangential speed based on distance
      // Use Rankine vortex model: v = vmax * (r/R) for r < R, v = vmax * (R/r)
      // for r > R
      double tangentialSpeed = 0.0;
      if (distance < 1e-6) {
        tangentialSpeed = 0.0; // Avoid division by zero at center
      } else if (distance <= radius) {
        tangentialSpeed = maxSpeedMps * (distance / radius);
      } else {
        tangentialSpeed = maxSpeedMps * (radius / distance);
      }

      // Convert to cartesian velocity components
      // For counterclockwise rotation: vx = -v_tangential * sin(θ), vy =
      // v_tangential * cos(θ)
      double theta = std::atan2(dy, dx);
      double vx = -tangentialSpeed * std::sin(theta);
      double vy = tangentialSpeed * std::cos(theta);

      cell.setVelocityX(vx);
      cell.setVelocityY(vy);
    }
  }
}

/**
 * @brief Apply a jet flow from a specified edge with unit conversion
 * @param edgePosition Edge position ('left', 'right', 'top', 'bottom')
 * @param centerPos Position along the edge for jet center
 * @param jetWidth Width of the jet in grid cells
 * @param jetSpeed Maximum jet speed in specified units
 * @param velocityUnit Velocity unit string (e.g., "m/s", "mph", "knot")
 */
void Grid::applyJetFlowWithUnits(const std::string &edgePosition,
                                 size_t centerPos, size_t jetWidth,
                                 double jetSpeed,
                                 const std::string &velocityUnit = "m/s") {
  double jetSpeedMps = Units::convert(jetSpeed, velocityUnit, "m/s");

  // Calculate half width of jet
  size_t halfWidth = jetWidth / 2;

  // Define ranges for the jet based on edge position
  size_t startI = 0, endI = 0, startJ = 0, endJ = 0;
  double vx = 0.0, vy = 0.0;

  if (edgePosition == "left") {
    startI = 0;
    endI = std::min(5, static_cast<int>(m_nx)); // First few columns
    startJ = (centerPos > halfWidth) ? centerPos - halfWidth : 0;
    endJ = std::min(centerPos + halfWidth, m_ny - 1);
    vx = jetSpeedMps;
    vy = 0.0;
  } else if (edgePosition == "right") {
    startI = m_nx > 5 ? m_nx - 5 : 0; // Last few columns
    endI = m_nx - 1;
    startJ = (centerPos > halfWidth) ? centerPos - halfWidth : 0;
    endJ = std::min(centerPos + halfWidth, m_ny - 1);
    vx = -jetSpeedMps;
    vy = 0.0;
  } else if (edgePosition == "top") {
    startI = (centerPos > halfWidth) ? centerPos - halfWidth : 0;
    endI = std::min(centerPos + halfWidth, m_nx - 1);
    startJ = 0;
    endJ = std::min(5, static_cast<int>(m_ny)); // First few rows
    vx = 0.0;
    vy = jetSpeedMps;
  } else if (edgePosition == "bottom") {
    startI = (centerPos > halfWidth) ? centerPos - halfWidth : 0;
    endI = std::min(centerPos + halfWidth, m_nx - 1);
    startJ = m_ny > 5 ? m_ny - 5 : 0; // Last few rows
    endJ = m_ny - 1;
    vx = 0.0;
    vy = -jetSpeedMps;
  }

  // Apply parabolic velocity profile across jet width
  for (size_t j = startJ; j <= endJ; ++j) {
    for (size_t i = startI; i <= endI; ++i) {
      Cell &cell = getCell(i, j);
      if (cell.isObstacle() || cell.isBoundary()) {
        continue;
      }

      // Calculate distance from center of jet
      double normalizedDist = 0.0;
      if (edgePosition == "left" || edgePosition == "right") {
        normalizedDist =
            std::abs(static_cast<double>(j) - centerPos) / halfWidth;
      } else {
        normalizedDist =
            std::abs(static_cast<double>(i) - centerPos) / halfWidth;
      }

      // Parabolic profile: v = vmax * (1 - (r/R)²)
      double factor = std::max(0.0, 1.0 - normalizedDist * normalizedDist);

      cell.setVelocityX(vx * factor);
      cell.setVelocityY(vy * factor);
    }
  }
}
