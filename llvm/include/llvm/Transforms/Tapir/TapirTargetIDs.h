//===- TapirTargetIDs.h - Tapir target ID's --------------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file enumerates the available Tapir lowering targets.
//
//===----------------------------------------------------------------------===//

#ifndef TAPIR_TARGET_IDS_H_
#define TAPIR_TARGET_IDS_H_

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

namespace llvm {

enum class TapirTargetID {
  Off,      // Completely disabled (i.e., no -ftapir argument was present).
  None,     // Perform no lowering
  Serial,   // Lower to serial projection
  Cuda,     // Lower to Cuda ABI  
  Hip,      // Lower to the Hip (AMDGPU) ABI.
  OpenCilk, // Lower to OpenCilk ABI
  OpenMP,   // Lower to OpenMP (TODO: Needs to be updated.)
  Qthreads, // Lower to Qthreads (TODO: Needs to be udpated.)
  Realm,    // Lower to Realm (TODO: Needs to be updated.)
  Last_TapirTargetID
};

enum class TapirNVArchTargetID {
  Off,      // Completely disabled (i.e., -ftapir != gpu|cuda)
  SM_50,    // TODO: Remove depcreated targets based on latest CUDA releases.
  SM_52,
  SM_53,
  SM_60,    // Pascal
  SM_61,
  SM_62,
  SM_70,    // Volta
  SM_72,
  SM_75,    // Turing
  SM_80,    // Ampere
  SM_86,   
  SM_90,    // Hopper
  // TODO: Update this enum when we sync w/ upstream LLVM capabilities.
  Last_TapirNVArchTargetID
};

// Tapir target options

// Virtual base class for Target-specific options.
class TapirTargetOptions {
public:
  enum TapirTargetOptionKind { TTO_OpenCilk, Last_TTO };

private:
  const TapirTargetOptionKind Kind;

public:
  TapirTargetOptionKind getKind() const { return Kind; }

  TapirTargetOptions(TapirTargetOptionKind K) : Kind(K) {}
  TapirTargetOptions(const TapirTargetOptions &) = delete;
  TapirTargetOptions &operator=(const TapirTargetOptions &) = delete;
  virtual ~TapirTargetOptions() {}

  // Top-level method for cloning TapirTargetOptions.  Defined in
  // TargetLibraryInfo.
  TapirTargetOptions *clone() const;
};

// Options for OpenCilkABI Tapir target.
class OpenCilkABIOptions : public TapirTargetOptions {
  std::string RuntimeBCPath;

  OpenCilkABIOptions() = delete;

public:
  OpenCilkABIOptions(StringRef Path)
      : TapirTargetOptions(TTO_OpenCilk), RuntimeBCPath(Path) {}

  StringRef getRuntimeBCPath() const {
    return RuntimeBCPath;
  }

  static bool classof(const TapirTargetOptions *TTO) {
    return TTO->getKind() == TTO_OpenCilk;
  }

protected:
  friend TapirTargetOptions;

  OpenCilkABIOptions *cloneImpl() const {
    return new OpenCilkABIOptions(RuntimeBCPath);
  }
};

} // end namespace llvm

#endif
