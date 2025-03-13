/**
 * @file Region.hpp
 * @brief Template method implementations for Region class
 * 
 * This file contains the implementation of template methods for the Region class.
 * It should be included after the Region.h header to ensure proper template
 * instantiation.
 */

#ifndef REGION_HPP
#define REGION_HPP

#include "RegionStorage.h"

// Template method implementations

template <typename LengthUnitT>
units::PhysicalBounds<LengthUnitT> Region::getPhysicalBounds() const {
    // Get cell index bounds from the region definition
    auto bounds = m_definition->getBounds();

    // Limit bounds to actual mesh size
    int minI = std::max(bounds.first.first, 0);
    int minJ = std::max(bounds.first.second, 0);
    int maxI = std::min(bounds.second.first, m_mesh->nx() - 1);
    int maxJ = std::min(bounds.second.second, m_mesh->ny() - 1);

    // Convert to physical positions with requested unit type
    units::Vector2D<LengthUnitT> minPos = m_mesh->template cellToPosition<LengthUnitT>(minI, minJ);
    units::Vector2D<LengthUnitT> maxPos = m_mesh->template cellToPosition<LengthUnitT>(maxI, maxJ);

    // Create physical bounds
    return units::PhysicalBounds<LengthUnitT>(minPos.x(), minPos.y(), maxPos.x(), maxPos.y());
}

template <typename LengthUnitT>
bool Region::containsPosition(const units::Vector2D<LengthUnitT>& position) const {
    // Convert physical position to cell indices
    auto [i, j] = m_mesh->positionToCell(position);

    // Check if the cell at these indices is in the region
    return containsIndex(m_mesh->linearIndex(i, j));
}

template <typename LengthUnitT>
Region Region::createFromPhysicalBounds(
    RegionID id,
    Mesh* mesh,
    const units::Vector2D<LengthUnitT>& minPos,
    const units::Vector2D<LengthUnitT>& maxPos,
    const std::string& name)
{
    if (!mesh) {
        throw std::invalid_argument("Cannot create region: mesh cannot be null");
    }

    // Convert physical positions to cell indices
    auto [minI, minJ] = mesh->positionToCell(minPos);
    auto [maxI, maxJ] = mesh->positionToCell(maxPos);

    // Create rectangular region definition
    auto definition = std::make_shared<RectangularRegion>(name, minI, minJ, maxI, maxJ);

    // Create region with mesh binding
    return Region(id, definition, mesh);
}

template <typename PosUnitT, typename RadiusUnitT>
Region Region::createPhysicalCircle(
    RegionID id,
    Mesh* mesh,
    const units::Vector2D<PosUnitT>& center,
    RadiusUnitT radius,
    const std::string& name)
{
    if (!mesh) {
        throw std::invalid_argument("Cannot create circle region: mesh cannot be null");
    }

    // Convert all units to meters (canonical unit) for internal calculations
    auto centerMeters = center.template as<units::meter>();
    units::meter radiusMeters = units::unit_cast<units::meter>(radius);

    // Define a predicate for cells within the circle
    auto circlePredicate = [centerMeters, radiusMeters, mesh](const Cell* cell) -> bool {
        if (!cell) return false;

        // Get physical position of cell center in meters
        auto cellPos = cell->getPosition<units::meter>();

        // Calculate squared distance to center (more efficient than using sqrt)
        double dx = cellPos.x().value() - centerMeters.x().value();
        double dy = cellPos.y().value() - centerMeters.y().value();
        double distSquared = dx * dx + dy * dy;

        // Check if within radius
        return distSquared <= radiusMeters.value() * radiusMeters.value();
    };

    // Create a predicate-based region definition
    auto definition = std::make_shared<PredicateRegion>(name,
        std::function<bool(const Cell*)>(circlePredicate));

    // Create region with mesh binding
    return Region(id, definition, mesh);
}

template <typename LengthUnitT, typename FuncT>
void Region::forEachCellWithPosition(FuncT func) const {
    // Ensure we have a valid mesh
    if (!m_mesh) {
        throw std::logic_error("Cannot iterate: no mesh bound to region");
    }

    // Choose the most efficient iteration strategy based on storage mode
    if (m_mode == RegionStorageMode::CELL_SET) {
        // Iterate through explicit cell indices
        for (int idx : m_cellIndices) {
            // Convert linear index to 2D coordinates
            auto [i, j] = m_mesh->toIndices(idx);

            // Get the cell and its position
            Cell* cell = m_mesh->getCell(i, j);
            if (cell) {
                units::Vector2D<LengthUnitT> pos = cell->template getPosition<LengthUnitT>();
                func(cell, pos);
            }
        }
    }
    else if (m_mode == RegionStorageMode::BIT_ARRAY) {
        // Iterate through bit array using efficient findFirst/findNext
        for (size_t idx = m_bitArray.findFirst(); idx < m_bitArray.size(); idx = m_bitArray.findNext(idx)) {
            // Convert linear index to 2D coordinates
            auto [i, j] = m_mesh->toIndices(static_cast<int>(idx));

            // Get the cell and its position
            Cell* cell = m_mesh->getCell(i, j);
            if (cell) {
                units::Vector2D<LengthUnitT> pos = cell->template getPosition<LengthUnitT>();
                func(cell, pos);
            }
        }
    }
    else if (m_mode == RegionStorageMode::DYNAMIC) {
        // For dynamic regions, iterate through all cells in the mesh bounds
        auto bounds = m_definition->getBounds();

        // Limit bounds to actual mesh size
        int minI = std::max(bounds.first.first, 0);
        int minJ = std::max(bounds.first.second, 0);
        int maxI = std::min(bounds.second.first, m_mesh->nx() - 1);
        int maxJ = std::min(bounds.second.second, m_mesh->ny() - 1);

        // Iterate through bounded region
        for (int j = minJ; j <= maxJ; ++j) {
            for (int i = minI; i <= maxI; ++i) {
                Cell* cell = m_mesh->getCell(i, j);
                if (cell && m_definition->contains(cell)) {
                    units::Vector2D<LengthUnitT> pos = cell->template getPosition<LengthUnitT>();
                    func(cell, pos);
                }
            }
        }
    }
}

template <typename LengthUnitT>
Region Region::dilateByDistance(LengthUnitT distance) const {
    // Ensure we have a valid mesh
    if (!m_mesh) {
        throw std::logic_error("Cannot dilate: no mesh bound to region");
    }

    // Convert to meters (canonical unit) for internal calculations
    units::meter distanceMeters = units::unit_cast<units::meter>(distance);

    // Convert physical distance to cell count
    // Use the smaller of dx and dy to ensure we cover at least the requested distance
    double cellSize = std::min(m_mesh->template dx<units::meter>().value(), 
                              m_mesh->template dy<units::meter>().value());
    int cellCount = static_cast<int>(std::ceil(distanceMeters.value() / cellSize));

    // Create a new cell set for the dilated region
    std::unordered_set<int> dilatedIndices;

    // Start with the original region's cells
    const auto& originalIndices = getCellIndices();
    dilatedIndices.insert(originalIndices.begin(), originalIndices.end());

    // For each cell in the original region, add cells within the distance
    for (int idx : originalIndices) {
        // Convert linear index to 2D coordinates
        auto [i, j] = m_mesh->toIndices(idx);

        // Iterate through surrounding cells within cell count
        for (int dj = -cellCount; dj <= cellCount; ++dj) {
            for (int di = -cellCount; di <= cellCount; ++di) {
                int ni = i + di;
                int nj = j + dj;

                // Skip if out of bounds
                if (ni < 0 || ni >= m_mesh->nx() || nj < 0 || nj >= m_mesh->ny()) {
                    continue;
                }

                // Calculate physical distance to original cell
                auto origPos = m_mesh->template cellToPosition<units::meter>(i, j);
                auto neighPos = m_mesh->template cellToPosition<units::meter>(ni, nj);
                
                // Use Vector2D's distanceTo method if available, otherwise calculate manually
                double dx = origPos.x().value() - neighPos.x().value();
                double dy = origPos.y().value() - neighPos.y().value();
                units::meter physDist = units::meter(std::sqrt(dx*dx + dy*dy));

                // Add if within the requested distance
                if (physDist.value() <= distanceMeters.value()) {
                    int linearIdx = m_mesh->linearIndex(ni, nj);
                    dilatedIndices.insert(linearIdx);
                }
            }
        }
    }

    // Create a new region from the dilated indices
    RegionID newId = m_id + 1; // Simple ID generation

    // Create a predicate-based definition for the dilated region
    auto predicate = [dilatedIndices, this](const Cell* cell) -> bool {
        if (!cell) return false;
        return dilatedIndices.find(cell->linearIndex()) != dilatedIndices.end();
    };

    auto definition = std::make_shared<PredicateRegion>(
        "Dilated_" + m_definition->getName(),
        std::function<bool(const Cell*)>(predicate));

    // Create the dilated region
    Region dilatedRegion(newId, definition, m_mesh);

    // Initialize with the dilated indices
    dilatedRegion.setStorageMode(RegionStorageMode::CELL_SET);
    auto& cellIndices = dilatedRegion.getCellIndicesForWrite();
    cellIndices = std::move(dilatedIndices);

    // Optimize storage based on new density
    dilatedRegion.optimizeStorage();

    return dilatedRegion;
}

template <typename LengthUnitT>
double Region::physicalArea() const {
    // Ensure we have a valid mesh
    if (!m_mesh) {
        throw std::logic_error("Cannot calculate area: no mesh bound to region");
    }

    // Get cell area in requested physical units
    double cellArea = m_mesh->template dx<LengthUnitT>().value() * 
                     m_mesh->template dy<LengthUnitT>().value();

    // Multiply by the number of cells in the region
    return cellArea * static_cast<double>(size());
}

#endif // REGION_HPP


