// Copyright 2016 Guiming Zhang (gzhang45@wisc.edu)
// Distributed under GNU General Public License (GPL) license

/*
* Kernel function
*/

#ifndef _KDE_KERNEL_H_
#define _KDE_KERNEL_H_

#include <stdio.h>
#include <stdlib.h>
#include <math_functions.h>

#include "SamplePoints.h"
#include "AsciiRaster.h"
#include "Utilities.h"

__device__ float dReductionSum = 1.0f; // sum of log of densities
__device__ float dDen0_0 = 1.0f; // sum of log of densities

// squared distance btw two points
__device__  float dDistance2(float x0, float y0, float x1, float y1){
	float dx = x1 - x0;
	float dy = y1 - y0;
	return dx*dx + dy*dy;
}

// Gaussian kernel
__device__ float dGaussianKernel(float h2, float d2){
	return expf(d2 / (-2.0f * h2)) / (h2 * TWO_PI);
}

// Edge correction with fixed bandwidth h2 (squared)
__global__ void CalcEdgeCorrectionWeights(float h2, const SamplePoints dPoints, const AsciiRaster dAscii, float* dWeights)
{

	// serial point ID
	int tid = (blockIdx.y * gridDim.y + blockIdx.x) * blockDim.x + threadIdx.x;

	// directly return if ID goes out of range
	if(tid >= dPoints.numberOfPoints){
		return;
	}

	// otherwise calculate edge effect correction weight point ID = tid
	float cellSize = dAscii.cellSize;
	int nCols = dAscii.nCols;
	int nRows = dAscii.nRows;
	float xLLCorner = dAscii.xLLCorner;
	float yLLCorner = dAscii.yLLCorner;
	float noDataValue = dAscii.noDataValue;

	//printf("%d %d\n", nCols, nRows);

	float cellArea = cellSize * cellSize;

	float p_x = dPoints.xCoordinates[tid];
	float p_y = dPoints.yCoordinates[tid];
	float ew = 0.0f;

	//printf("%d %.3f %.3f\n", tid, p_x, p_y);

	float cell_x, cell_y, val, d2;//, g;
	int row, col;
	for (row = 0; row < nRows; row++){
		for (col = 0; col < nCols; col++){
			val = dAscii.elements[row*nCols+col];
			if (val != noDataValue){
				cell_x = COL_TO_XCOORD(col, xLLCorner, cellSize);
				cell_y = ROW_TO_YCOORD(row, nRows, yLLCorner, cellSize);
				d2 = dDistance2(p_x, p_y, cell_x, cell_y);
				//if(d2 < 9.0 * h2){
					ew += dGaussianKernel(h2, d2) * cellArea;
				//}
			}
		}
	}
	dWeights[tid] = 1.0f / ew;
}

// Edge correction with adaptive bandwidth (variable bandwidth at each point in dHs)
__global__ void CalcEdgeCorrectionWeights(float* dHs, const SamplePoints dPoints, const AsciiRaster dAscii, float* dWeights)
{

	// serial point ID
	int tid = (blockIdx.y * gridDim.y + blockIdx.x) * blockDim.x + threadIdx.x;

	// directly return if ID goes out of range
	if(tid >= dPoints.numberOfPoints){
		return;
	}

	// otherwise calculate edge effect correction weight point ID = tid
	float cellSize = dAscii.cellSize;
	int nCols = dAscii.nCols;
	int nRows = dAscii.nRows;
	float xLLCorner = dAscii.xLLCorner;
	float yLLCorner = dAscii.yLLCorner;
	float noDataValue = dAscii.noDataValue;

	//printf("%d %d\n", nCols, nRows);

	float cellArea = cellSize * cellSize;

	float p_x = dPoints.xCoordinates[tid];
	float p_y = dPoints.yCoordinates[tid];
	float ew = 0.0f;
	float h = dHs[tid];
	float h2 = h * h;

	//printf("%d %.3f %.3f\n", tid, p_x, p_y);

	float cell_x, cell_y, val, d2;//, g;
	int row, col;
	for (row = 0; row < nRows; row++){
		for (col = 0; col < nCols; col++){
			val = dAscii.elements[row*nCols+col];
			if (val != noDataValue){
				cell_x = COL_TO_XCOORD(col, xLLCorner, cellSize);
				cell_y = ROW_TO_YCOORD(row, nRows, yLLCorner, cellSize);
				d2 = dDistance2(p_x, p_y, cell_x, cell_y);

				//if(d2 < 9.0 * h2){
					ew += dGaussianKernel(h2, d2) * cellArea;
				//}
			}
		}
	}
	dWeights[tid] = 1.0f / ew;
}


// Kernel density estimation with fixed bandwidth h2 (squared)
__global__ void KernelDesityEstimation(float h2, const SamplePoints dPoints, const AsciiRaster dAscii, float* dWeights)
{
	// serial point ID
	int tid = (blockIdx.y * gridDim.y + blockIdx.x) * blockDim.x + threadIdx.x;

	// # of rows and cols
	int nCols = dAscii.nCols;
	int nRows = dAscii.nRows;

	// directly return if ID goes out of range
	if(tid >= nCols * nRows){
		return;
	}

	// otherwise, do KDE
	float cellSize = dAscii.cellSize;
	float xLLCorner = dAscii.xLLCorner;
	float yLLCorner = dAscii.yLLCorner;
	float noDataValue = dAscii.noDataValue;
	float cell_x, cell_y; // x,y coord of cell
	float p_x, p_y, p_w;    // x, y coord, weight of point
	int numPoints = dPoints.numberOfPoints;
	float d2;
	float e_w = 1.0;    // edge effect correction weight
	float den;
	int col, row;

	// which row, col?
	row = tid / nCols;
	col = tid - row * nCols;

	// x, y coord of this cell
	cell_y = ROW_TO_YCOORD(row, nRows, yLLCorner, cellSize);
	cell_x = COL_TO_XCOORD(col, xLLCorner, cellSize);

	// should do KDE on this cell?
	float val = dAscii.elements[tid];

	if(val == noDataValue) {
		return;
	}

	den = 0.0f;
	for (int p = 0; p < numPoints; p++){
		p_x = dPoints.xCoordinates[p];
		p_y = dPoints.yCoordinates[p];
		p_w = dPoints.weights[p];
		e_w = dWeights[p];
		d2 = dDistance2(p_x, p_y, cell_x, cell_y);

		//if(d2 < 9.0 * h2){
			den += dGaussianKernel(h2, d2) * p_w *e_w;
		//}
	}
	dAscii.elements[tid] = den; // intensity, not probability
}

// Kernel density estimation with adaptive bandwidth (variable bandwidth at each point in dHs)
__global__ void KernelDesityEstimation(float* dHs, const SamplePoints dPoints, const AsciiRaster dAscii, float* dWeights)
{
	// serial point ID
	int tid = (blockIdx.y * gridDim.y + blockIdx.x) * blockDim.x + threadIdx.x;

	// # of rows and cols
	int nCols = dAscii.nCols;
	int nRows = dAscii.nRows;

	// directly return if ID goes out of range
	if(tid >= nCols * nRows){
		return;
	}

	// otherwise, do KDE
	float cellSize = dAscii.cellSize;
	float xLLCorner = dAscii.xLLCorner;
	float yLLCorner = dAscii.yLLCorner;
	float noDataValue = dAscii.noDataValue;
	float cell_x, cell_y; // x,y coord of cell
	float p_x, p_y, p_w;    // x, y coord, weight of point
	int numPoints = dPoints.numberOfPoints;
	float h, d2;
	float e_w = 1.0f;    // edge effect correction weight
	float den;
	int col, row;

	// which row, col?
	row = tid / nCols;
	col = tid - row * nCols;

	// x, y coord of this cell
	cell_y = ROW_TO_YCOORD(row, nRows, yLLCorner, cellSize);
	cell_x = COL_TO_XCOORD(col, xLLCorner, cellSize);

	// should do KDE on this cell?
	float val = dAscii.elements[tid];

	if(val == noDataValue) {
		return;
	}

	den = 0.0f;
	for (int p = 0; p < numPoints; p++){
		p_x = dPoints.xCoordinates[p];
		p_y = dPoints.yCoordinates[p];
		p_w = dPoints.weights[p];
		e_w = dWeights[p];
		h = dHs[p];
		d2 = dDistance2(p_x, p_y, cell_x, cell_y);

		//if(d2 < 9.0 * h * h){
			den += dGaussianKernel(h * h, d2) * p_w *e_w;
		//}

		//den += dGaussianKernel(h * h, d2) * p_w *e_w;
	}
	dAscii.elements[tid] = den; // intensity, not probability
}

// Density at each point under fixed bandwidth h2 (squared)
__global__ void DensityAtPoints(float h2, const SamplePoints dPoints, float *dWeights, float* dDen0, float* dDen1){
	// serial point ID
	int tid = (blockIdx.y * gridDim.y + blockIdx.x) * blockDim.x + threadIdx.x;

	int n = dPoints.numberOfPoints;
	// directly return if ID goes out of range
	if(tid >= n){
		return;
	}

	// otherwise calculate density at point ID = tid
	float p_x = dPoints.xCoordinates[tid];
	float p_y = dPoints.yCoordinates[tid];
	float den = 0.0f;

	//printf("%d %.3f %.3f\n", tid, p_x, p_y);
	int i;
	float x, y, p_w, e_w, d2;//, g;
	float den_itselft = 0.0f;
	for (i = 0; i < n; i++){
		x = dPoints.xCoordinates[i];
		y = dPoints.yCoordinates[i];
		p_w = dPoints.weights[i];
		e_w = dWeights[i];
		d2 = dDistance2(p_x, p_y, x, y);

		//if(d2 < 9.0 * h2){
			den += dGaussianKernel(h2, d2) * p_w *e_w;
		//}

		//g = dGaussianKernel(h2, d2) * p_w *e_w;
		//den += g;
	}

	x = dPoints.xCoordinates[tid];
	y = dPoints.yCoordinates[tid];
	p_w = dPoints.weights[tid];
	e_w = dWeights[tid];
	d2 = dDistance2(p_x, p_y, x, y);
	//if(d2 < 9.0 * h2){
		den_itselft = dGaussianKernel(h2, d2) * p_w *e_w;
	//}

	if(dDen0 != NULL){
		//dDen0[tid] = den;
		dDen0[tid] = logf(den);
	}

	if(dDen1 != NULL){
		//dDen1[tid] = den - den_itselft;
		dDen1[tid] = logf(den - den_itselft);
	}
}

// Density at each point under adaptive bandwidth (variable bandwidth at each point in dHs)
__global__ void DensityAtPoints(float* dHs, const SamplePoints dPoints, float *dWeights, float* dDen0, float* dDen1){
	// serial point ID
	int tid = (blockIdx.y * gridDim.y + blockIdx.x) * blockDim.x + threadIdx.x;

	int n = dPoints.numberOfPoints;
	// directly return if ID goes out of range
	if(tid >= n){
		return;
	}
	//printf("dHS[%d]: %.2f\n", tid, dHs[tid]);

	// otherwise calculate density at point ID = tid
	float p_x = dPoints.xCoordinates[tid];
	float p_y = dPoints.yCoordinates[tid];
	float den = 0.0f;

	//printf("%d %.3f %.3f\n", tid, p_x, p_y);
	int i;
	float x, y, h, p_w, e_w, d2;//, g;
	float den_itselft = 0.0f;
	for (i = 0; i < n; i++){
		x = dPoints.xCoordinates[i];
		y = dPoints.yCoordinates[i];
		p_w = dPoints.weights[i];
		e_w = dWeights[i];
		d2 = dDistance2(p_x, p_y, x, y);
		h = dHs[i];
		//if(d2 < 9.0 * h * h){
			den += dGaussianKernel(h * h, d2) * p_w *e_w;
		//}

		//den += g;

		//printf("%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n", x, y, p_w, e_w, d2, h, g, den);
	}



	x = dPoints.xCoordinates[tid];
	y = dPoints.yCoordinates[tid];
	p_w = dPoints.weights[tid];
	e_w = dWeights[tid];
	d2 = dDistance2(p_x, p_y, x, y);
	h = dHs[tid];
	//if(d2 < 9.0 * h * h){
		den_itselft = dGaussianKernel(h * h, d2) * p_w *e_w;
	//}

	//printf("%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n", x, y, p_w, e_w, d2, h, g, den);

	if(dDen0 != NULL){
		//dDen0[tid] = den;
		dDen0[tid] = logf(den);
		//printf("dDen0[%d]: %.5f\n", tid, dDen0[tid]);
	}

	if(dDen1 != NULL){
		//dDen1[tid] = den - den_itselft;
		dDen1[tid] = logf(den - den_itselft);
		//printf("dDen1[%d]: %.5f\n", tid, dDen1[tid]);
	}
}

// compute spatially varying bandwidths
__global__ void CalcVaryingBandwidths(const SamplePoints dPoints, float h, float * dHs)
{

	// serial point ID
	int tid = (blockIdx.y * gridDim.y + blockIdx.x) * blockDim.x + threadIdx.x;

	// directly return if ID goes out of range
	int n = dPoints.numberOfPoints;
	if(tid >= n){
		return;
	}

	// otherwise calculate varying bandwidth for point ID = tid
	dHs[tid] = h;
}

// compute spatially varying bandwidths
__global__ void CalcVaryingBandwidths(const SamplePoints dPoints, float* dDen0, float h, float alpha, float * dHs)
{

	// serial point ID
	int tid = (blockIdx.y * gridDim.y + blockIdx.x) * blockDim.x + threadIdx.x;

	// directly return if ID goes out of range
	int n = dPoints.numberOfPoints;
	if(tid >= n){
		return;
	}

	// otherwise calculate varying bandwidth for point ID = tid
	float g = expf(dReductionSum / n);
	float den = dDen0[tid];
	//if(tid == 0){
	//	den = dDen0_0;
	//}
	//float tmph = (h * (powf(expf(den) / g, alpha))); // this outmost () is NECESSARY!
	dHs[tid] = (h * (powf(expf(den) / g, alpha)));
	//printf("dHs[%d]: %4.5f \n", tid, dHs[tid]);
	//dHs[tid] = h;
}

// **===----------------- Parallel reduction (sum) ---------------------===**
//! @param g_data           input array in global memory
//                          result is expected in index 0 of g_idata
//! @param N                input number of elements to scan from input data
//! @param iteration        current iteration in reduction
// **===------------------------------------------------------------------===**
__global__ void ReductionSum(float *g_data, unsigned int N, int iteration, int num_active_items)
{
	// use shared memory
	__shared__ float s_data[BLOCK_SIZE];

	unsigned int thread_id = threadIdx.x;
	unsigned int serial_thread_id = (blockIdx.y * gridDim.y + blockIdx.x) * blockDim.x + threadIdx.x;
	// each thread loads one element from global to shared memory
	unsigned int arrIdx =  serial_thread_id * powf(BLOCK_SIZE, iteration);

	if(arrIdx < N && serial_thread_id < num_active_items){
		s_data[thread_id] = g_data[arrIdx];
	}
	else{
		s_data[thread_id] = 0.0f;
	}

	// sync threads to ensure all data are loaded into shared memory
	__syncthreads();

	// # of elements in the array to reduce
	unsigned int n_ele = BLOCK_SIZE; // initial # of elements = 1024

	// recursively reduce the array
	while(n_ele > 1){
		unsigned int m = n_ele / 2;
		if(thread_id < m){
			s_data[thread_id] += s_data[thread_id + m];
		}
		__syncthreads();
		n_ele /= 2;
	}

	// write result back to global memory
	if(thread_id == 0){
		unsigned int idx = (blockIdx.y * gridDim.y + blockIdx.x) * blockDim.x * powf(blockDim.x, iteration);
		if(idx < N){
			/*if(idx == 0){
				dDen0_0 = g_data[idx];
			}*/
			g_data[idx] = s_data[0];
		}


		if(num_active_items <= BLOCK_SIZE){
			dReductionSum = g_data[0];
		}
	}

}

#endif // #ifndef _KDE_KERNEL_H_
