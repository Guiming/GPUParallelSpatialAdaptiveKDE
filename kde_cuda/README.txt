Requires cmake to compile using CMakeLists.txt (this helps set up the environment needed).

You may need to change lines #92 and #93 in file "ParallelUtils.cmake" in accordance with the NVIDIA GPU in use (i.e., compute capability version):
--------------
# select Compute Capability
# This needs to be manually updated when devices with new CCs come out
set(CUDA_DEVICE_VERSION "20" CACHE STRING "CUDA Device Version")
set_property(CACHE CUDA_DEVICE_VERSION PROPERTY STRINGS "10" "11" "12" "13"	"20" "21" "30" "32" "35" "37" "50" "52")
---------------
For example, on a Ubuntu OS with Quadro P4000 (compute capability 6.1 according to https://en.wikipedia.org/wiki/CUDA), they are changed to: 
---------------
set(CUDA_DEVICE_VERSION "61" CACHE STRING "CUDA Device Version")
set_property(CACHE CUDA_DEVICE_VERSION PROPERTY STRINGS "10" "11" "12" "13"	"20" "21" "30" "32" "35" "37" "50" "52" "53" "60" "61" "62" "70" "72" "75" "80" "86")
---------------

Then run the following to compile the program:
cmake CMakeLists.txt
make

Three versions of the algorithm are included:
#1 without any optimization
#2 partial optimization (avoiding re-computing edge correction factors)
#3 full optimization (avoiding re-computing edge correction factors and kd-tree indexing)

After compiling, run the program by:

./PROGRM 1 POINTS_FILE STUDY_AREA_MASK_FILE BANDWIDTH_OPTION SKIP_SEQUENTIAL_COMPUTATION SKIP_PARALLEL_COMPUTATION OUTPUT_DENSITY_FILE_SEQUENTIAL  OUTPUT_DENSITY_FILE_PARALLEL >> LOG_FILE

BANDWIDTH_OPTION: 0 - rule of thumb; 1 - cross validation (fixed); 2 - adaptive

e.g.,
./kde_cuda_kdtr 1 pntsRedwood.csv redwood.asc 0 1 0 redwood_SEQ.asc redwood_GPU.asc >> redwood_LOG.log
