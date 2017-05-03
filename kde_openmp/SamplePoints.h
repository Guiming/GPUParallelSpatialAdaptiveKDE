// Copyright 2016 Guiming Zhang (gzhang45@wisc.edu)
// Distributed under GNU General Public License (GPL) license

#ifndef _SAMPLEPOINTS_H_
#define _SAMPLEPOINTS_H_


// SamplePoints Structure declaration
typedef struct {
	unsigned int numberOfPoints;
	float* xCoordinates;
	float* yCoordinates;
	float* weights;
	float* distances; // closest distances (squared) to study area boundary (by Guiming @ 2016-09-01)
} SamplePoints;


#endif // _SAMPLEPOINTS_H_