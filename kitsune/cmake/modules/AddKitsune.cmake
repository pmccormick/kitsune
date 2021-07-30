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
  set(_rt_cmake_var_names "OPENMP;QTHREADS;REALM;CUDATK;HIP;OPENCL")
  # OpenCilk runtime target (a.k.a., cheetah) is always enabled. 
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

