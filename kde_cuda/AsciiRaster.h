// Copyright 2016 Guiming Zhang (gzhang45@wisc.edu)
// Distributed under GNU General Public License (GPL) license

#ifndef _ASCIIRASTER_H_
#define _ASCIIRASTER_H_


// AsciiRaster Structure declaration
typedef struct {
	unsigned int nCols;
	unsigned int nRows;
	float xLLCorner;
	float yLLCorner;
	float cellSize;
	float noDataValue;
	float* elements;
} AsciiRaster;


#endif // _ASCIIRASTER_H_