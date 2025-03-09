/**
 * @file Mesh.hpp
 * @brief Template implementation of the Mesh class
 * 
 * This file contains the implementation of template methods
 * declared in Mesh.h. It is included at the end of Mesh.h
 * and should not be included directly.
 */

#include <stdexcept>
#include <typeinfo>
#include <typeindex>

template <typename CellType>
Mesh<CellType>::Mesh(int nx, int ny, double dx, double dy, double originX, double originY)
    : MeshBase(nx, ny, dx, dy), m_originX(originX), m_originY(originY) 
{
    // Allocate storage for cells
    m_cells.resize(nx * ny);
    
    // Initialize cells
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            m_cells[linearIndex(i, j)] = CellType(i, j, this);
        }
    }
}

template <typename CellType>
std::pair<double, double> Mesh<CellType>::getPhysicalDimensions() const {
    return {m_nx * m_dx, m_ny * m_dy};
}

template <typename CellType>
std::pair<double, double> Mesh<CellType>::cellToPhysical(int i, int j) const {
    return {m_originX + i * m_dx, m_originY + j * m_dy};
}

template <typename CellType>
std::pair<int, int> Mesh<CellType>::physicalToCell(double x, double y) const {
    int i = static_cast<int>((x - m_originX) / m_dx);
    int j = static_cast<int>((y - m_originY) / m_dy);
    return {i, j};
}

template <typename CellType>
CellBase* Mesh<CellType>::getCell(int i, int j) {
    if (i >= 0 && i < m_nx && j >= 0 && j < m_ny) {
        return &m_cells[linearIndex(i, j)];
    }
    return nullptr;
}

template <typename CellType>
CellType* Mesh<CellType>::getTypedCell(int i, int j) {
    if (i >= 0 && i < m_nx && j >= 0 && j < m_ny) {
        return &m_cells[linearIndex(i, j)];
    }
    return nullptr;
}

template <typename CellType>
CellType* Mesh<CellType>::getNeighborCell(int i, int j, uint8_t direction) const {
    // Calculate neighbor indices based on direction
    auto [di, dj] = CellBase::getDirectionOffset(direction);
    int ni = i + di;
    int nj = j + dj;
    
    // Check if the neighbor is within bounds
    if (ni >= 0 && ni < m_nx && nj >= 0 && nj < m_ny) {
        return &m_cells[linearIndex(ni, nj)];
    }
    
    return nullptr;
}

template <typename CellType>
template <typename T, typename LocationTag>
Field<T, LocationTag>& Mesh<CellType>::createField(FieldID fieldID, 
                                               const std::string& name,
                                               const T& defaultValue) {
    // Check if field already exists
    if (hasField(fieldID)) {
        throw std::runtime_error("Field with ID " + std::to_string(fieldID) + " already exists");
    }
    
    // Create field entry
    auto entry = std::make_unique<FieldEntry>();
    entry->typeName = typeid(Field<T, LocationTag>).name();
    entry->name = name.empty() ? "Field_" + std::to_string(fieldID) : name;
    entry->elemSize = sizeof(T);
    entry->mask = static_cast<FieldMask>(1) << (fieldID % 64); // Use bit position as mask
    
    // Create the field
    auto field = new Field<T, LocationTag>(m_nx, m_ny);
    entry->fieldPtr = field;
    
    // Initialize with default value if provided
    field->fill(defaultValue);
    
    // Store the field entry
    auto& result = *field;
    m_fields[fieldID] = std::move(entry);
    
    return result;
}

template <typename CellType>
template <typename T, typename LocationTag>
Field<T, LocationTag>& Mesh<CellType>::getField(FieldID fieldID) {
    auto it = m_fields.find(fieldID);
    if (it == m_fields.end()) {
        throw std::runtime_error("Field with ID " + std::to_string(fieldID) + " does not exist");
    }
    
    const auto& entry = it->second;
    if (!validateFieldType<T, LocationTag>(entry.get())) {
        throw std::runtime_error("Type mismatch for field with ID " + std::to_string(fieldID));
    }
    
    return *static_cast<Field<T, LocationTag>*>(entry->fieldPtr);
}

template <typename CellType>
template <typename T, typename LocationTag>
const Field<T, LocationTag>& Mesh<CellType>::getField(FieldID fieldID) const {
    auto it = m_fields.find(fieldID);
    if (it == m_fields.end()) {
        throw std::runtime_error("Field with ID " + std::to_string(fieldID) + " does not exist");
    }
    
    const auto& entry = it->second;
    if (!validateFieldType<T, LocationTag>(entry.get())) {
        throw std::runtime_error("Type mismatch for field with ID " + std::to_string(fieldID));
    }
    
    return *static_cast<Field<T, LocationTag>*>(entry->fieldPtr);
}

template <typename CellType>
bool Mesh<CellType>::hasField(FieldID fieldID) const {
    return m_fields.find(fieldID) != m_fields.end();
}

template <typename CellType>
std::pair<double, double> Mesh<CellType>::getOrigin() const {
    return {m_originX, m_originY};
}

template <typename CellType>
FieldMask Mesh<CellType>::getFieldMask(FieldID fieldID) const {
    auto it = m_fields.find(fieldID);
    if (it == m_fields.end()) {
        return 0; // No mask if field doesn't exist
    }
    
    return it->second->mask;
}

template <typename CellType>
RegionMask Mesh<CellType>::getRegionMask(RegionID regionID) const {
    auto it = m_regionMasks.find(regionID);
    if (it == m_regionMasks.end()) {
        return 0; // No mask if region doesn't exist
    }
    
    return it->second;
}

template <typename CellType>
template <typename T, typename LocationTag>
bool Mesh<CellType>::validateFieldType(const FieldEntry* entry) const {
    if (!entry) return false;
    
    // Check element size
    if (entry->elemSize != sizeof(T)) {
        return false;
    }
    
    // Check type name (may be platform-dependent, but provides an additional check)
    const std::string& expected = typeid(Field<T, LocationTag>).name();
    return entry->typeName == expected;
}

// Implementation of FieldEntry destructor to properly delete the type-erased field
template <typename CellType>
Mesh<CellType>::FieldEntry::~FieldEntry() {
    // The field pointer needs to be deleted, but we need to know its type
    // This will be handled by the specialized Mesh classes in generated code
    if (fieldPtr) {
        // In a real implementation, we would need to store the field type
        // information to properly delete the object here
        // For now, we'll assume nothing needs to be done (memory leak)
        // Generated code will handle this correctly
    }
}


