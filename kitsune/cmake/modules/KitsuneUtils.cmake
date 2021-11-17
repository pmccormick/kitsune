#
# Get a list of all enabled tapir runtime targets so we can
# walk through each and do "stuff" (e.g., build an executable
# for each enabled target from a single source file).
#
# NOTE: This implementation assumes there is a one-to-one
# match between the compiler option flag and the name used
# in the CMake variables.  In other words,
#
#   -ftapir=TARGET_NAME
#
# has a corresponding CMAKE configuration flag:
#
#   KITSUNE_ENABLE_(TARGET_NAME)_TARGET
#
# In addition, the list '_rt_cmake_var_names' below will need
# to be updated to include any new runtime targets added to
# Kitsune+Tapir.
#
macro(get_tapir_rt_targets rt_list)
  set(_rt_cmake_var_names "OPENMP;QTHREADS;REALM;CUDA;HIP;OPENCL")
  # Always enable the serial and opencilk targets.
  list(APPEND ${rt_list} "serial")
  list(APPEND ${rt_list} "opencilk")

  foreach(rt IN ITEMS ${_rt_cmake_var_names})
    set(_enabled_var "KITSUNE_ENABLE_${rt}_TARGET")
    message(STATUS "checking for ${_enabled_var}")
    if (${_enabled_var})
      string(TOLOWER ${rt} flag)
      list(APPEND ${rt_list} ${flag})
    endif()
  endforeach()
  unset(_rt_cmake_var_names)
  unset(_kitsune_rt_flags)
endmacro()

# NOTE: the dep_list and target_libs variables used in the following
# function are expected to live in the parent scope (i.e., we're
# mucking with state outside of the function to try and debug some
# build dependency issues).
function(add_tapir_dependency target abi)
  message(STATUS "adding dependency for ${target} w/ -ftapir=${abi}")
  if (${abi} STREQUAL "opencilk")
    list(APPEND dep_list cheetah)
    #set(target_libs ${target_libs} opencilk opencilk-personality-cpp PARENT_SCOPE)
    add_dependencies(${target} ${dep_list})
  elseif(${abi} STREQUAL "realm")
    list(APPEND dep_list RealmABI)
    set(target_libs ${target_libs} realm-abi realm PARENT_SCOPE)
    add_dependencies(${target} ${dep_list})
  elseif(${abi} STREQUAL "cuda")
    list(APPEND dep_list cu-abi)
    set(target_libs ${target_libs} cu-abi PARENT_SCOPE)
    add_dependencies(${target} ${dep_list})
  elseif(${abi} STREQUAL "none")
    message(STATUS "no dependencies for '-ftapir=none'...")
  elseif(${abi} STREQUAL "serial")
    message(STATUS "no dependencies for '-ftapir=serial'...")
  else()
    message(FATAL_ERROR
       "tapir dependency ${abi} not handled in add_tapir_dependency")
  endif()

endfunction()
