# This module will set the following variables in your project:
# 
#  CilkRTS_FOUND        -- cilk runtime was found.
#  CilkRTS_INCLUDE_DIR  -- directory with cilk runtime header file(s). 
#  CilkRTS_LIBRARY      -- full path to cilk runtime library. 
#  CilkRTS_LIBRARY_DIR  -- path to where the cilk runtime library is installed. 
#  CilkRTS_LINK_LIBS    -- set of link libraries (e.g. -lcilkrts) 
# 

message(STATUS "Looking for cilkrts...")

find_path(CilkRTS_INCLUDE_DIR cilk/cilk.h
  PATHS /usr/local/include
        /opt/include
        /opt/local/include
)

find_library(CilkRTS_LIBRARY cilkrts
  PATHS /usr/local/lib64
        /usr/local/lib
        /opt/lib64
        /opt/lib	
        /opt/local/lib64
        /opt/local/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CilkRTS DEFAULT_MSG
				  CilkRTS_INCLUDE_DIR 
				  CilkRTS_LIBRARY)


if (CilkRTS_FOUND) 
  message(STATUS "Found Cilk runtime.")
  get_filename_component(CilkRTS_LIBRARY_DIR
                         ${CilkRTS_LIBRARY}
                         DIRECTORY
                         CACHE)
			 
  set(CilkRTS_LINK_LIBS "-lcilkrts" 
      CACHE STRING "List of libraries needed for Cilk runtime.")
      
  message(STATUS "Cilk runtime include directory : ${CilkRTS_INCLUDE_DIR}")
  message(STATUS "Cilk runtime library directory : ${CilkRTS_LIBRARY_DIR}")
  message(STATUS "Cilk runtime link libraries    : ${CilkRTS_LINK_LIBS}")
  
else()

  message(STATUS "Cilk rutnime NOT found.")
  

endif()

