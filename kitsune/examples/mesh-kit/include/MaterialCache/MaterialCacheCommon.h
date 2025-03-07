/**
 * @file MaterialCacheCommon.h
 * @brief Common include file for the Material Cache system
 * @details
 *
 * This file provides a single include point for all Material Cache
 * components, making it easier to use the caching system in client code.
 */

#pragma once

// Core caching system
#include "MaterialCacheCore.h"
#include "MaterialCacheStatistics.h"

// Main cache components
#include "MaterialCache.h"
#include "MaterialCacheFactory.h"
#include "SpecializedCaches.h"

// Equivalence and eviction strategies
#include "EquivalenceKeys.h"
#include "EvictionPolicies.h"

// Enhanced statistics and contamination detection (optional)
#include "ContaminationReport.h"
#include "EnhancedStatistics.h"
#include "MaterialBehaviorTracker.h"

/**
 * @namespace MaterialCache
 * @brief Namespace for the Material Cache system
 *
 * This namespace contains all components of the Material Cache system,
 * providing a unified interface for caching, strategy selection, and analysis.
 */
namespace MC {
// Type aliases for convenience
using Cache = MaterialCache;
using FixedCache = FixedSizeMaterialCache;
using AdaptiveCache = AdaptiveMaterialCache;

// Equivalence key aliases
using StandardKey = StandardEquivalenceKey;
using AdaptiveKey = AdaptiveEquivalenceKey;
using DomainKey = DomainSpecificEquivalenceKey;

// Eviction policy aliases
using LRUPolicy = LRUEvictionPolicy;
using MRUPolicy = MRUEvictionPolicy;
using FIFOPolicy = FIFOEvictionPolicy;
using FrequencyPolicy = FrequencyEvictionPolicy;
using TimeSensitivePolicy = TimeSensitiveEvictionPolicy;
using AdaptivePolicy = AdaptiveEvictionPolicy;
} // namespace MC
