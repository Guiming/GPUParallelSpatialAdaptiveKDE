Requires cmake to compile using CMakeLists.txt (this helps set up the environment needed).
Three versions of the algorithm are included:
#1 without any optimization
#2 partial optimization (avoiding re-computing edge correction factors)
#3 full optimization (avoiding re-computing edge correction factors and kd-tree indexing)