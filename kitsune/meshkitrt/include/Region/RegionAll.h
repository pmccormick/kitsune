/**
 * @file RegionAll.h
 * @brief Convenience header that includes all Region system components
 * 
 * This header simplifies using the Region system by including all necessary
 * components in the correct inclusion order. Including this file provides
 * access to all Region classes, operations, and utilities.
 */

#ifndef REGION_ALL_H
#define REGION_ALL_H

// Core components
#include "BitArray.h"
#include "Region.h"
#include "RegionDefinition.h"

// Region definitions and utilities
#include "RegionUtils.h"

// Storage and optimization components
#include "RegionStorage.h"
#include "RegionOptimization.h"

// Operation components
#include "RegionSetOperations.h"
#include "RegionAccessor.h"

#endif // REGION_ALL_H

