####################################################
##   Only modify if you know what you're doing.   ##
####################################################


# Helps Eclipse/CDT find our include directories
set(CMAKE_VERBOSE_MAKEFILE on)

# Detect the bitness of our machine (eg 32- or 64-bit)
# C-equiv: sizeof(void*)
# Alt: 8*sizeof(void*)
math(EXPR CMAKE_ARCH_BITNESS 8*${CMAKE_SIZEOF_VOID_P})

# For non-multi-configuration generators (eg, make, Eclipse)
# The Visual Studio generator creates a single project with all these
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "For single-configuration generators (e.g. make) set the type of build: Release, Debug, RelWithDebugInfo, MinSizeRel")
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Release" "Debug" "RelWithDebugInfo" "MinSizeRel")


####################################################
## ---------------------------------------------- ##
## -                                            - ##
## -            Enable MPI Support              - ##
## -                                            - ##
## ---------------------------------------------- ##
####################################################

# Begin configuring MPI options
macro(enable_mpi_support)

    find_package("MPI" REQUIRED)

    # Add the MPI-specific compiler and linker flags
    # Also, search for #includes in MPI's paths

    set(CMAKE_C_COMPILE_FLAGS "${CMAKE_C_COMPILE_FLAGS} ${MPI_C_COMPILE_FLAGS}")
    set(CMAKE_C_LINK_FLAGS "${CMAKE_C_LINK_FLAGS} ${MPI_C_LINK_FLAGS}")
    include_directories(${MPI_C_INCLUDE_PATH})

    set(CMAKE_CXX_COMPILE_FLAGS "${CMAKE_CXX_COMPILE_FLAGS} ${MPI_CXX_COMPILE_FLAGS}")
    set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} ${MPI_CXX_LINK_FLAGS}")
    include_directories(${MPI_CXX_INCLUDE_PATH})

endmacro(enable_mpi_support)
# Done configuring MPI Options


####################################################
## ---------------------------------------------- ##
## -                                            - ##
## -            Enable OpenMP Support           - ##
## -                                            - ##
## ---------------------------------------------- ##
####################################################

# Begin configuring OpenMP options
macro(enable_openmp_support)

    find_package("OpenMP" REQUIRED)

    # Add the OpenMP-specific compiler and linker flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")

endmacro(enable_openmp_support)
# Done configuring OpenMP Options


####################################################
## ---------------------------------------------- ##
## -                                            - ##
## -            Enable CUDA Support             - ##
## -                                            - ##
## ---------------------------------------------- ##
####################################################

# Begin configuring CUDA options
# This is ugly...
macro(enable_cuda_support)

    # Hide a number of options from the default CMake screen
    mark_as_advanced(CLEAR CUDA_BUILD_CUBIN)
    mark_as_advanced(CLEAR CUDA_TOOLKIT_ROOT_DIR)
    mark_as_advanced(CLEAR CUDA_VERBOSE_BUILD)
    mark_as_advanced(CLEAR CUDA_FAST_MATH)
    mark_as_advanced(CLEAR CUDA_USE_CUSTOM_COMPILER)
    mark_as_advanced(CLEAR CUDA_VERBOSE_PTX)
    mark_as_advanced(CLEAR CUDA_DEVICE_VERSION)

    # select Compute Capability
    # This needs to be manually updated when devices with new CCs come out
    set(CUDA_DEVICE_VERSION "20" CACHE STRING "CUDA Device Version")
    set_property(CACHE CUDA_DEVICE_VERSION PROPERTY STRINGS "10" "11" "12" "13"	"20" "21" "30" "32" "35" "37" "50" "52")

    # Enable fast-math for CUDA (_not_ GCC)
    set(CUDA_FAST_MATH TRUE CACHE BOOL "Use Fast Math Operations")

    # Tell nvcc to use a separate compiler for non-CUDA code.
    # This is useful if you need to use an older of GCC than comes by default
    set(CUDA_USE_CUSTOM_COMPILER FALSE CACHE BOOL "Use Custom Compiler")
    set(CUDA_CUSTOM_COMPILER "" CACHE STRING "Custom C++ Compiler for CUDA If Needed")

    # Shows register usage, etc
    set(CUDA_VERBOSE_PTX TRUE CACHE BOOL "Show Verbose Kernel Info During Compilation")


    # Let's get going...
    find_package("CUDA" REQUIRED)

    # Set custom compiler flags
    set(CUDA_NVCC_FLAGS "" CACHE STRING "" FORCE)

    if(CUDA_USE_CUSTOM_COMPILER)
        mark_as_advanced(CLEAR CUDA_CUSTOM_COMPILER)
        list(APPEND CUDA_NVCC_FLAGS "-ccbin=${CUDA_CUSTOM_COMPILER}")
    else()
        mark_as_advanced(FORCE CUDA_CUSTOM_COMPILER)
    endif()

    # Macro for setting the Compute Capability
    macro(set_compute_capability cc)
        list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_${cc},code=sm_${cc}")
        list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_${cc},code=compute_${cc}")
    endmacro(set_compute_capability)

    # Tell nvcc to compile for the selected Compute Capability
    # This can also be called from the main CMakeLists.txt to enable
    # support for additional CCs
    set_compute_capability(${CUDA_DEVICE_VERSION})

    # Enable fast-math if selected
    if(CUDA_FAST_MATH)
        list(APPEND CUDA_NVCC_FLAGS "-use_fast_math")
    endif()

    # Enable verbose compile if selected
    if(CUDA_VERBOSE_PTX)
        list(APPEND CUDA_NVCC_FLAGS "--ptxas-options=-v")
    endif()
endmacro(enable_cuda_support)
# Done configuring CUDA options
