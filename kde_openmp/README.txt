Requires cmake to compile using CMakeLists.txt (this helps set up the environment needed).
Three versions of the algorithm are included:
#1 without any optimization
#2 partial optimization (avoiding re-computing edge correction factors)
#3 full optimization (avoiding re-computing edge correction factors and kd-tree indexing)

After compiling, run the program by:

./PROGRM 1 POINTS_FILE STUDY_AREA_MASK_FILE BANDWIDTH_OPTION NUMBER_OF_THREADS SKIP_SEQUENTIAL_COMPUTATION SKIP_PARALLEL_COMPUTATION OUTPUT_DENSITY_FILE_SEQUENTIAL  OUTPUT_DENSITY_FILE_PARALLEL >> LOG_FILE

BANDWIDTH_OPTION: 0 - rule of thumb; 1 - cross validation (fixed); 2 - adaptive

e.g.,
./kde_omp_kdtr 1 pntsRedwood.csv redwood.asc 2 32 1 0 redwood_SEQ.asc redwood_OMP.asc >> redwood_LOG.log