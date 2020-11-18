# This module will set the following variables in your project:
# 
#  OpenCilk_FOUND        -- opencilk runtime was found.
#  OpenCilk_INCLUDE_DIR  -- directory with opencilk runtime header file(s). 
#  OpenCilk_LIBRARY      -- full path to opencilk runtime library. 
#  OpenCilk_LIBRARY_DIR  -- path to where the opencilk runtime library lives.
#  OpenCilk_LINK_LIBS    -- set of link libraries (e.g. -lopencilk) 
# 

message(STATUS "Looking for open cilk...")

find_path(OpenCilk_INCLUDE_DIR cilk/cilk.h
  PATHS /usr/local/include
        /opt/include
        /opt/local/include
)

find_library(OpenCilk_LIBRARY opencilk
  PATHS /usr/local/lib64
        /usr/local/lib
        /opt/lib64
        /opt/lib	
        /opt/local/lib64
        /opt/local/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenCilk DEFAULT_MSG
				  OpenCilk_INCLUDE_DIR 
				  OpenCilk_LIBRARY)


if (OpenCilk_FOUND) 
  message(STATUS "Found OpenCilk runtime.")
  get_filename_component(OpenCilk_LIBRARY_DIR
                         ${OpenCilk_LIBRARY}
                         DIRECTORY
                         CACHE)
			 
  set(OpenCilk_LINK_LIBS "-lopencilk" 
      CACHE STRING "List of libraries needed for OpenCilk runtime.")
      
  message(STATUS "OpenCilk runtime include directory : ${OpenCilk_INCLUDE_DIR}")
  message(STATUS "OpenCilk runtime library directory : ${OpenCilk_LIBRARY_DIR}")
  message(STATUS "OpenCilk runtime link libraries    : ${OpenCilk_LINK_LIBS}")
else()
  message(STATUS "OpenCilk rutnime NOT found.")
endif()

