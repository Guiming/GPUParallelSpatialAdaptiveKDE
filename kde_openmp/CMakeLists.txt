# Minimum version of CMake required.
cmake_minimum_required(VERSION 2.8)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
# Set the name of your project
project(KDEPROJECT)

# Include macros from the ParallelUtils library
include(ParallelUtils.cmake)

##########################################################
# If using Makefiles, be sure to set CMAKE_BUILD_TYPE to #
# select different build configurations.                 #
# Possible values:                                       #
# - RELEASE - fully optimized for speed                  #
# - DEBUG - unoptimized, with debugging symbols          #
# - RELWITHDEBUGINFO - optimized, with debugging symbols #
# - MINSIZEREL - optimized for size                      #
# - [not set] - unoptimized                              #
##########################################################


## Example MPI program
#enable_mpi_support()
#add_executable(integrate_mpi integrate_mpi.cpp)
#target_link_libraries(integrate_mpi ${MPI_CXX_LIBRARIES})

## Example OpenMP program
enable_openmp_support()
#1 without any optimization
add_executable(kde_omp_old kde_omp_old.cpp)
add_executable(kde_omp_old_sequential kde_omp_old_sequential.cpp)
#2 partial optimization (avoiding re-computing edge correction factors)
add_executable(kde_omp kde_omp.cpp)
add_executable(kde_omp_sequential kde_omp_sequential.cpp)
#3 full optimization (avoiding re-computing edge correction factors and kd-tree indexing)
add_executable(kde_omp_kdtr kde_omp_kdtr.cpp KDtree.cpp)
add_executable(kde_omp_kdtr_sequential kde_omp_kdtr_sequential.cpp KDtree.cpp)

set_target_properties(kde_omp PROPERTIES COMPILE_FLAGS "-o4")
set_target_properties(kde_omp_sequential PROPERTIES COMPILE_FLAGS "-o4")
set_target_properties(kde_omp_old PROPERTIES COMPILE_FLAGS "-o4")
set_target_properties(kde_omp_old_sequential PROPERTIES COMPILE_FLAGS "-o4")
set_target_properties(kde_omp_kdtr PROPERTIES COMPILE_FLAGS "-o4")
set_target_properties(kde_omp_kdtr_sequential PROPERTIES COMPILE_FLAGS "-o4")

## compile option for profiling
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
#SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
#SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")
