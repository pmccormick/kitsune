
/**
 * @file AccessorIterators.h
 * @brief Iterator interfaces for Accessor classes
 * 
 * This file defines iterator interfaces for the various accessor classes
 * (MeshAccessor, RegionAccessor, FieldAccessor, CompoundAccessor).
 * These iterators build on the Field iterators to provide consistent and
 * efficient traversal patterns for mesh data.
 */

#ifndef ACCESSOR_ITERATORS_H
#define ACCESSOR_ITERATORS_H

#include "FieldIterators.h"
#include <functional>

// Forward declarations
template <typename T, typename LocationTag> class Field;
using RegionMask = uint64_t;
using FieldMask = uint64_t;
using FieldID = uint32_t;

#include "AccessorIteratorsCommon.h"
#include "MeshAccessorIterators.h"
#include "RegionAccessorIterators.h"
#include "FieldAccessorIterators.h"
#include "CompoundAccessorIterators.h"

#endif // ACCESSOR_ITERATORS_H
 
