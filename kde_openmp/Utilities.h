// Copyright 2016 Guiming Zhang (gzhang45@wisc.edu)
// Distributed under GNU General Public License (GPL) license

/*
  utilies
*/
#define TWO_PI 6.2831853071795862f
#define PI 3.1415926535897931f
#define SQRT_PI 1.772453851f
#define FLOAT_MAX 99999999999999.0f

#define OMP_SCHEDULE_PNTS dynamic // static // guided
#define OMP_SCHEDULE_CELLS static // dynamic // guided

/*
#define ROW_TO_YCOORD(ROW, NROWS, YLLCORNER, CELLSIZE) YLLCORNER + (NROWS - 0.5 - ROW) * CELLSIZE
#define COL_TO_XCOORD(COL, XLLCORNER, CELLSIZE) XLLCORNER + (COL + 0.5) * CELLSIZE

#define YCOORD_TO_ROW(Y, NROWS, YLLCORNER, CELLSIZE) NROWS - 1 - (int)((Y - YLLCORNER) / CELLSIZE)
#define XCOORD_TO_COL(X, XLLCORNER, CELLSIZE) (int)((X - XLLCORNER) / CELLSIZE)


#define MIN(X, Y) X < Y ? X : Y
#define MAX(X, Y) X < Y ? Y : X

*/
#define TIMEDIFF(T2,T1) chrono::duration_cast<chrono::milliseconds>(T2 - T1).count()

#define EPSILONDENSITY 0.0f

#define UPDATEWEIGHTS true

#define DEBUG true
#define BRUNSDON false

//#define cut-off factor, default value 9.0
#define CUT_OFF_FACTOR 9.0
#define SQRT_CUT_OFF_FACTOR 3.0


#define NARROW true
#define MAX_TREE_LEVEL 7

#define ATOMIC true
#define MAX_NUM_ITERATIONS 30
