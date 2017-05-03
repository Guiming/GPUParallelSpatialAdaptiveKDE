// Copyright 2016 Guiming Zhang (gzhang45@wisc.edu)
// Distributed under GNU General Public License (GPL) license

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <float.h>

#include "SamplePoints.h"
#include "AsciiRaster.h"
#include "Utilities.h"
#include "KDtree.h"

using namespace std;


/* ONLY FOR PROFILING WITH GPROF*/
inline float ROW_TO_YCOORD(int ROW, int NROWS, float YLLCORNER, float CELLSIZE) {return YLLCORNER + (NROWS - 0.5 - ROW) * CELLSIZE;}
inline float COL_TO_XCOORD(int COL, float XLLCORNER, float CELLSIZE) {return XLLCORNER + (COL + 0.5) * CELLSIZE;}

inline int YCOORD_TO_ROW(float Y, int NROWS, float YLLCORNER, float CELLSIZE) {return NROWS - 1 - (int)((Y - YLLCORNER) / CELLSIZE);}
inline int XCOORD_TO_COL(float X, float XLLCORNER, float CELLSIZE) {return (int)((X - XLLCORNER) / CELLSIZE);}

inline int MIN(int X, int Y){return X < Y ? X : Y;}
inline int MAX(int X, int Y){return X < Y ? Y : X;}
/* END */

// distance squared between two points
inline  float Distance2(float x0, float y0, float x1, float y1){
	float dx = x1 - x0;
	float dy = y1 - y0;
	return dx*dx + dy*dy;
}

// mean center of points
void MeanCenter(SamplePoints Points, float &mean_x, float &mean_y);

// (squared) standard distance of points
void StandardDistance2(SamplePoints Points, float &d2);

// bandwidth squared
inline float BandWidth2(SamplePoints Points){
	float d2;
	StandardDistance2(Points, d2);
	return sqrtf(2.0 / (3 * Points.numberOfPoints)) * d2;
}

// Gaussian kernel
inline float GaussianKernel(float h2, float d2){
	if(d2 >= CUT_OFF_FACTOR * h2){
		return 0.0f;
	}

	if(BRUNSDON){
		return expf(-1.0 * d2 / h2) / (h2 * SQRT_PI); // Brunsdon 1995 parameter
		//return exp(d2 / (-2.0 * h2)) / (h2*TWO_PI);
	}
	else{
		//return exp(d2 / (-2.0 * h2)) / (h2*PI);
		return expf(d2 / (-2.0 * h2)) / (h2*TWO_PI); // Gaussian
		//return 0.75 * (1.0 - (d2/h2)*(d2/h2)); // Epanechnikov
	}
}

// compute the optimal Maximum Likelihood Estimation fixed bandwidth
// By Guiming @ 2016-02-26
float MLE_FixedBandWidth(AsciiRaster Ascii, SamplePoints Points, float *edgeWeights, float h, int nThreads = 1);

// compute the log likelihood given bandwidth h2
// By Guiming @ 2016-02-26
float LogLikelihood(AsciiRaster Ascii, SamplePoints Points, float *edgeWeights, float h, bool update = true, int nThreads = 1);

// compute the log likelihood given adaptive bandwidths hs
// By Guiming @ 2016-02-26
// float* den0 : density based on all points, including itself
// float* den1 : leave one out density
float LogLikelihoodAdaptive(AsciiRaster Ascii, SamplePoints Points, float *edgeWeights, float* hs, bool update = true, float* den0 = NULL, float* den1 = NULL, int nThreads = 1);

// compute the log likelihood given a center (h0, alpha0) and step (stepH, stepA)
// By Guiming @ 2016-03-06
void hj_likelihood(AsciiRaster Ascii, SamplePoints Points, float *edgeWeights, float h0, float alpha0, float stepH, float stepA, int lastdmax, float* logLs, int nThreads = 1);

// compute the optimal h and alpha (parameters for calculating the optimal adaptive bandwith)
// By Guiming @ 2016-03-06
void hooke_jeeves(AsciiRaster Ascii, SamplePoints Points, float *edgeWeights, float h0, float alpha0, float stepH, float stepA, float* optParas, int nThreads = 1);

float compGML(float* den0, int numPoints);

SamplePoints AllocateSamplePoints(int n); // random points
SamplePoints ReadSamplePoints(const char *csvFile); // points read from a .csv file
// By Guiming @ 2016-09-04
SamplePoints CopySamplePoints(const SamplePoints Points); // copy points
void FreeSamplePoints(SamplePoints* Points);
void WriteSamplePoints(SamplePoints* Points, const char * csvFile);
void WriteSamplePoints(SamplePoints* Points, float* Hs, float* Ws, const char * csvFile);


AsciiRaster AllocateAsciiRaster(int nCols, int nRows, float xLLCorner, float yLLCorner, float cellSize, float noDataValue);
AsciiRaster ReadAsciiRaster(char * asciiFile); // ascii raster read from a .asc file
AsciiRaster CopyAsciiRaster(const AsciiRaster Ascii);
void FreeAsciiRaster(AsciiRaster* Ascii);
void WriteAsciiRaster(AsciiRaster* Ascii, const char * asciiFile);

float* AllocateEdgeCorrectionWeights(SamplePoints Points);
void FreeEdgeCorrectionWeights(float* weights);

float* AllocateBandWidths(SamplePoints Points);
void FreeBandWidths(float* hs);

float* AllocateDen(SamplePoints Points); // n is number of points
void FreeDen(float* den);


// exact edge effects correction (Diggle 1985) (parallel)
void EdgeCorrectionWeightsExact(SamplePoints Points, float *hs, AsciiRaster Ascii, float *weights, int nThreads = 1);
void EdgeCorrectionWeightsExact(SamplePoints Points, float h, AsciiRaster Ascii, float *weights, int nThreads = 1);

// exact edge effects correction (Diggle 1985) (sequential)
//void EdgeCorrectionWeightsExact(SamplePoints Points, float *hs, AsciiRaster Ascii, float *weights);

// check whether the result from sequential computation and that from parallel computation agree
void CheckResults(AsciiRaster AsciiSEQ, AsciiRaster AsciiPARA);

// extract study area boundary from a raster
// By Guiming @ 2016-09-01
void MarkBoundary(AsciiRaster Ascii, int nThreads = 1);

// compute the closest distances from sample points to study area boundary
// By Guiming @ 2016-09-01
void CalcDist2Boundary(SamplePoints Points, AsciiRaster Ascii, int nThreads = 1);

// sort the sample points on their distances to study area boundary
// By Guiming @ 2016-09-04
void SortSamplePoints(SamplePoints Points);

// comparison function for sort
// By Guiming @ 2016-09-04
int compare ( const void *pa, const void *pb );



/* Run in 2 modes
 *
 * Mode 0: Do not read points and mask from files.
 *         User specify # of points and cell size of the estimated intensity surface.
 *         Random points with x, y coordinates in the range [0,100] will be generated.
 *         The cell size (must be less than 100) determines how many cells in the intensity surface raster.
 *
 *         ./kde_omp [mode] [#points] [cellsize] [#threads] [skipSEQ] [skipPARA]
 *         e.g., ./kde_omp 0 100 1.0 4 0 0
 *
 * Mode 1: Read points and mask from files.
 * 					./kde_omp 1 data/Points.csv data/Mask.asc 0 1 0 1 seq.asc omp.asc
 *
*/

int main(int argc, char *argv[]){

	int NUM_THREADS = 1;
	int NPNTS = 100;                // default # of points
	float CELLSIZE = 1.0f;          // default cellsize
	char* pntFn = "data/Points.csv";  // default points file
	char* maskFn = "data/Mask.asc";   // default mask file
	bool fromFiles = true;          // by default, read Points and Mask from files

	int SKIPSEQ = 0;               // skip executing sequential program
	int SKIPPARA = 0;              // skip executing parallel program

	//Guiming May 1, 2016
	int Hoption = 0; // 0 for rule of thumb
					 // 1 for h optimal
					 // 2 for h adaptive
	char* denSEQfn = "data/den_SEQ.asc";
	char* denOMPfn = "data/den_OMP.asc";

	// parse commandline arguments
	if(argc != 10){
		printf("Incorrect arguments provided. Exiting...\n");
		printf("Run in mode 0:\n ./kde_omp 0 #points cellsize h_option #threads skip_sequential skip_parallel denfn_seq, denfn_omp\n");
		printf("Run in mode 1:\n ./kde_omp 1 points_file mask_file h_option #threads skip_sequential skip_parallel denfn_seq, denfn_omp\n");
        return 1;
	}
	else{
		int mode = atoi(argv[1]);
		if(mode == 0){
			fromFiles = false;
			NPNTS = atoi(argv[2]);
			CELLSIZE = (float)atof(argv[3]);
			Hoption = atoi(argv[4]);
			NUM_THREADS = atoi(argv[5]);
			SKIPSEQ = atoi(argv[6]);
			SKIPPARA = atoi(argv[7]);
			denSEQfn = argv[8];
			denOMPfn = argv[9];
		}
		else if(mode == 1){
			pntFn = argv[2];
			maskFn = argv[3];
			Hoption = atoi(argv[4]);
			NUM_THREADS = atoi(argv[5]);
			SKIPSEQ = atoi(argv[6]);
			SKIPPARA = atoi(argv[7]);
			denSEQfn = argv[8];
			denOMPfn = argv[9];
		}
		else{
			printf("Incorrect arguments provided. Exiting...\n");
			printf("Run in mode 0:\n ./kde_omp 0 #points cellsize h_option #threads skip_sequential skip_parallel denfn_seq, denfn_omp\n");
			printf("Run in mode 1:\n ./kde_omp 1 points_file mask_file h_option #threads skip_sequential skip_parallel denfn_seq, denfn_omp\n");
	        return 1;
		}

	}

	SamplePoints Points;   // sample of point events
	AsciiRaster Mask;      // a mask indicating the extent of study area
	AsciiRaster DenSurf, DenSurf_OMP;   // the estimated intensity surface
	float *edgeWeights;    // edge effect correct weights (for each point in the sample)
	float *bandWidths; // band width (for each point in the sample)

	bool correction = true;  // enable edge effect correction


	srand(100);  // If not read from files, generate random points

	if (fromFiles){
		Points = ReadSamplePoints(pntFn);
		Mask = ReadAsciiRaster(maskFn);
	}
	else{
		Points = AllocateSamplePoints(NPNTS);
		Mask = AllocateAsciiRaster(int(100/CELLSIZE), int(100/CELLSIZE), 0.0f, 0.0f, CELLSIZE, -9999.0f);

		WriteSamplePoints(&Points, "pntsSim.csv");
		WriteAsciiRaster(&Mask, "maskSim.asc");
	}

	// By Guiming @ 2016-09-01
	/*
	double stt = omp_get_wtime();
	CalcDist2Boundary(Points, Mask, 4);
	double stp = omp_get_wtime();
	printf("#calculating distance on %d points took %f ms\n", Points.numberOfPoints, (stp - stt) * 1000);
	stt = omp_get_wtime();
	SortSamplePoints(Points);
	stp = omp_get_wtime();
	printf("#sorting %d points took %f ms\n", Points.numberOfPoints, (stp - stt) * 1000);
	*/

	DenSurf = CopyAsciiRaster(Mask);

	// parameters
	int numPoints = Points.numberOfPoints;
	int nCols = Mask.nCols;
	int nRows = Mask.nRows;
	float xLLCorner = Mask.xLLCorner;
	float yLLCorner = Mask.yLLCorner;
	float noDataValue = Mask.noDataValue;
	float cellSize = Mask.cellSize;

	printf("number of points: %d\n", numPoints);
	printf("cell size: %f\n", cellSize);
	printf("number of cells: %d\n", nCols * nRows);

	printf("skip executing SEQUENTIAL program? %d\n", SKIPSEQ);
	printf("skip executing PARALLEL program? %d\n", SKIPPARA);
	printf("number of CPU cores: %d\n", NUM_THREADS);

	// do the work
	float cell_x; // x coord of cell
	float cell_y; // y coord of cell
	float p_x;    // x coord of point
	float p_y;    // x coord of point
	float p_w;    // weight of point
	float e_w = 1.0;    // edge effect correction weight

	// bandwidth (squared) for KDE
	float h = sqrtf(BandWidth2(Points));
	printf("rule of thumb bandwidth h0: %.5f\n", h);

	// timing
	double start, stop;
	float elaps_seq, elaps_para;

	if(SKIPSEQ == 0){

		edgeWeights = NULL;
		edgeWeights = AllocateEdgeCorrectionWeights(Points);

		bandWidths = NULL;
		bandWidths = AllocateBandWidths(Points);
		for(int i = 0; i < numPoints; i++){
			bandWidths[i] = h;
		}

		//////////////////////////////// START OF SEQUENTIAL ///////////////////////
		start = omp_get_wtime();

		// By Guiming @ 2016-09-11
		double start0 = start;
		MarkBoundary(Mask);
		printf("#marking boundary took %f ms\n", (omp_get_wtime() - start0) * 1000);
		// By Guiming @ 2016-09-11
		start0 = omp_get_wtime();
		CalcDist2Boundary(Points, Mask);
		printf("#computing distance took %f ms\n", (omp_get_wtime() - start0) * 1000);
		//WriteAsciiRaster(&Mask, "output/boundaryOMP.asc");
		start0 = omp_get_wtime();
		SortSamplePoints(Points);
		printf("#sorting distance took %f ms\n", (omp_get_wtime() - start0) * 1000);
		WriteSamplePoints(&Points, "pntsSim10000_sorted.csv");

		// compute edge effect correction weights (for each point) sequentially
		EdgeCorrectionWeightsExact(Points, bandWidths, Mask, edgeWeights, 1);

		if(Hoption == 1){
			float hopt = MLE_FixedBandWidth(Mask, Points, edgeWeights, h);
			printf("cross validated optimal fixed bandwidth hopt: %.5f\n", hopt);
			for(int i = 0; i < numPoints; i++){
				bandWidths[i] = hopt;
			}

			if(UPDATEWEIGHTS){
				EdgeCorrectionWeightsExact(Points, bandWidths, Mask, edgeWeights, 1);
			}
		}

		if(Hoption == 2){
			float h0 = h;
			float alpha0 = -0.5;
			float stepH = h0/10;
			float stepA = 0.1;
			float* optParas = (float*)malloc(3*sizeof(float));
			hooke_jeeves(Mask, Points, edgeWeights, h0, alpha0, stepH, stepA, optParas);
			h0 = optParas[0];
			alpha0 = optParas[1];
			float logL = optParas[2];

			if(DEBUG) printf("h0: %.5f alpha0: %.5f Lmax: %.5f\n", h0, alpha0, logL);

			free(optParas);
			optParas = NULL;

			float* den0 = AllocateDen(Points);
			for(int i = 0; i < numPoints; i++){
				bandWidths[i] = h0;
			}
			LogLikelihoodAdaptive(Mask, Points, edgeWeights, bandWidths, true, den0, NULL);
			float gml = compGML(den0, numPoints);
			for(int i = 0; i < numPoints; i++){
				bandWidths[i] = h0 * powf(den0[i]/gml, alpha0);
			}
			FreeDen(den0);

			if(UPDATEWEIGHTS){
				EdgeCorrectionWeightsExact(Points, bandWidths, Mask, edgeWeights, 1);
			}
		}

		//WriteSamplePoints(&Points, bandWidths, edgeWeights, "pntsSEQ.csv");

		// kernel density estimation
		for (int row = 0; row < nRows; row++){
			cell_y = ROW_TO_YCOORD(row, nRows, yLLCorner, cellSize);
			for (int col = 0; col < nCols; col++){
				cell_x = COL_TO_XCOORD(col, xLLCorner, cellSize);
				int idx = row * nCols + col;
				if (DenSurf.elements[idx] != noDataValue){

					float den = 0.0;
					for (int p = 0; p < numPoints; p++){
						p_x = Points.xCoordinates[p];
						p_y = Points.yCoordinates[p];
						p_w = Points.weights[p];
						if (correction){
							e_w = edgeWeights[p];
						}
						float d2 = Distance2(p_x, p_y, cell_x, cell_y);
						den += GaussianKernel(bandWidths[p] * bandWidths[p], d2) * p_w *e_w;
					}
					//den = den / numPoints;
					DenSurf.elements[idx] = den; // intensity, not probability
				}
			}
		}
		stop = omp_get_wtime();
		//////////////////////////////// END OF SEQUENTIAL ///////////////////////
		elaps_seq = (stop - start)*1000.f;
		printf("SEQUENTIAL computation took %.3f ms\n", elaps_seq);

		// write results to file
		WriteAsciiRaster(&DenSurf, denSEQfn);

		// clean up (only for those not needed any more)
		FreeEdgeCorrectionWeights(edgeWeights);
		FreeBandWidths(bandWidths);
		//FreeAsciiRaster(&DenSurf);
	}

//////////////////////////  START OF PARALLEL //////////////////////////
	if(SKIPPARA == 0){

		// By Guiming @ 2016-09-01
		//CalcDist2Boundary(Points, Mask, NUM_THREADS);
		//WriteAsciiRaster(&Mask, "output/boundaryOMP.asc");

		DenSurf_OMP = CopyAsciiRaster(Mask);

		if(NUM_THREADS > omp_get_max_threads()){
			NUM_THREADS = omp_get_max_threads();
		}
		edgeWeights = NULL;
		edgeWeights = AllocateEdgeCorrectionWeights(Points);

		bandWidths = NULL;
		bandWidths = AllocateBandWidths(Points);
		for(int i = 0; i < numPoints; i++){
			bandWidths[i] = h;
		}

		// start timing here
		start = omp_get_wtime();

		double start0 = start;
		MarkBoundary(Mask, NUM_THREADS);
		printf("#marking boundary took %f ms\n", (omp_get_wtime() - start0) * 1000);
		// By Guiming @ 2016-09-11
		start0 = omp_get_wtime();
		CalcDist2Boundary(Points, Mask, NUM_THREADS);
		printf("#computing distance took %f ms\n", (omp_get_wtime() - start0) * 1000);
		//WriteAsciiRaster(&Mask, "output/boundaryOMP.asc");
		start0 = omp_get_wtime();
		SortSamplePoints(Points);
		printf("#sorting distance took %f ms\n", (omp_get_wtime() - start0) * 1000);

		// compute edge effect correction weights in parallel
		EdgeCorrectionWeightsExact(Points, bandWidths, Mask, edgeWeights, NUM_THREADS);

		if(Hoption == 1){
			float hopt = MLE_FixedBandWidth(Mask, Points, edgeWeights, h, NUM_THREADS);
			printf("cross validated optimal fixed bandwidth hopt: %.5f\n", hopt);
			for(int i = 0; i < numPoints; i++){
				bandWidths[i] = hopt;
			}

			// update edge correction weights
			if(UPDATEWEIGHTS){
				EdgeCorrectionWeightsExact(Points, bandWidths, Mask, edgeWeights, NUM_THREADS);
			}
		}

		if(Hoption == 2){
			float h0 = h;
			float alpha0 = -0.5;
			float stepH = h0/10;
			float stepA = 0.1;

			// Brunsdon 1995 parameter settings
			if(BRUNSDON){
				h0 = 0.05;
				alpha0 = -1.5;
				stepH = 0.01;
				stepA = 0.1;
			}

			float* optParas = (float*)malloc(3*sizeof(float));
			hooke_jeeves(Mask, Points, edgeWeights, h0, alpha0, stepH, stepA, optParas, NUM_THREADS);
			h0 = optParas[0];
			alpha0 = optParas[1];
			float logL = optParas[2];

			if(DEBUG) printf("h0: %.5f alpha0: %.5f Lmax: %.5f\n", h0, alpha0, logL);

			free(optParas);
			optParas = NULL;

			float* den0 = AllocateDen(Points);
			for(int i = 0; i < numPoints; i++){
				bandWidths[i] = h0;
			}
			LogLikelihoodAdaptive(Mask, Points, edgeWeights, bandWidths, true, den0, NULL, NUM_THREADS);
			float gml = compGML(den0, numPoints);
			for(int i = 0; i < numPoints; i++){
				bandWidths[i] = h0 * powf(den0[i]/gml, alpha0);
			}
			FreeDen(den0);

			// update edge correction weights
			if(UPDATEWEIGHTS){
				EdgeCorrectionWeightsExact(Points, bandWidths, Mask, edgeWeights, NUM_THREADS);
			}
		}


	//WriteSamplePoints(&Points, bandWidths, edgeWeights, "data/ObsAptHsTRUE.csv");

	// KDE
	#pragma omp parallel for schedule(OMP_SCHEDULE_CELLS) num_threads(NUM_THREADS) private(p_x, p_y, cell_x, cell_y, p_w, e_w)
		for (int row = 0; row < nRows; row++){
			cell_y = ROW_TO_YCOORD(row, nRows, yLLCorner, cellSize);
			for (int col = 0; col < nCols; col++){
				cell_x = COL_TO_XCOORD(col, xLLCorner, cellSize);
				int idx = row * nCols + col;
				if (DenSurf_OMP.elements[idx] != noDataValue){

					float den = 0.0;
					for (int p = 0; p < numPoints; p++){
						p_x = Points.xCoordinates[p];
						p_y = Points.yCoordinates[p];
						p_w = Points.weights[p];
						if (correction){
							e_w = edgeWeights[p];
						}
						float d2 = Distance2(p_x, p_y, cell_x, cell_y);
						den += GaussianKernel(bandWidths[p] * bandWidths[p], d2) * p_w *e_w;
						//printf("den: %.6f\n", den);
					}
					//den = den / numPoints;
					DenSurf_OMP.elements[idx] = den; // intensity, not probability
				}
			}
		}

		// stop timing here
		stop = omp_get_wtime();

		elaps_para = (stop - start)*1000.f;

		printf("PARALLEL comptuation took %.3f ms on %d cores\n", elaps_para, NUM_THREADS);
		if(SKIPSEQ == 0) {
			printf("SPEEDUP: %.3fx\n", elaps_seq / elaps_para);
			// check the resutls
			CheckResults(DenSurf, DenSurf_OMP);
		}

		// write results to file
		WriteAsciiRaster(&DenSurf_OMP, denOMPfn);

		// clean up
		FreeSamplePoints(&Points);
		FreeEdgeCorrectionWeights(edgeWeights);
		FreeBandWidths(bandWidths);
		FreeAsciiRaster(&Mask);
		FreeAsciiRaster(&DenSurf);
		FreeAsciiRaster(&DenSurf_OMP);
	}
	/////////////////////////// END OF PARALLEL /////////////////////

	printf("Done...\n\n");

	return 0;
}

// mean center of points
void MeanCenter(SamplePoints Points, float &mean_x, float& mean_y){
	float sum_x = 0.0;
	float sum_y = 0.0;

	for (int p = 0; p < Points.numberOfPoints; p++){
		sum_x += Points.xCoordinates[p];
		sum_y += Points.yCoordinates[p];
	}

	mean_x = sum_x / Points.numberOfPoints;
	mean_y = sum_y / Points.numberOfPoints;
}

// standard distance squared
void StandardDistance2(SamplePoints Points, float &d2){

	float mean_x, mean_y;
	MeanCenter(Points, mean_x, mean_y);

	float sum2 = 0.0;

	for (int p = 0; p < Points.numberOfPoints; p++){
		sum2 += Distance2(mean_x, mean_y, Points.xCoordinates[p], Points.yCoordinates[p]);
	}

	d2 = sum2 / Points.numberOfPoints;
}

// generate random sample points
SamplePoints AllocateSamplePoints(int n){
	SamplePoints Points;

	Points.numberOfPoints = n;
	int size = Points.numberOfPoints;

	Points.xCoordinates = (float*)malloc(size*sizeof(float));
	Points.yCoordinates = (float*)malloc(size*sizeof(float));
	Points.weights = (float*)malloc(n*sizeof(float));
	Points.distances = (float*)malloc(n*sizeof(float));

	for (unsigned int i = 0; i < n; i++)
	{
		Points.xCoordinates[i] = (rand() / (float)RAND_MAX) * 100 ;
		Points.yCoordinates[i] = (rand() / (float)RAND_MAX) * 100 ;
		Points.weights[i] = 1.0f;
		Points.distances[i] = 0.0f;
	}
	return Points;
}

// points read from a .csv file
SamplePoints ReadSamplePoints(const char *csvFile){
	FILE *f = fopen(csvFile, "rt");
	if (f == NULL)
	{
		printf("Error opening file!\n");
		exit(1);
	}

	const int CSV_LINE_LENGTH = 256;
	SamplePoints Points;
	int n = 0;
	char line[CSV_LINE_LENGTH];
	char ch;

	while (!feof(f))
	{
		ch = fgetc(f);
		if (ch == '\n')
		{
			n++;
		}
	}

	if (n == 1){
		printf("No point in file!\n");
		exit(1);
	}

	n = n - 1; // do not count the header line
	Points.numberOfPoints = n;
	Points.xCoordinates = (float*)malloc(n*sizeof(float));
	Points.yCoordinates = (float*)malloc(n*sizeof(float));
	Points.weights = (float*)malloc(n*sizeof(float));
	Points.distances = (float*)malloc(n*sizeof(float));

	int counter = 0;
	char * pch;
	float x, y;
	rewind(f); // go back to the beginning of file
	fgets(line, CSV_LINE_LENGTH, f); //skip the header line
	while (fgets(line, CSV_LINE_LENGTH, f) != NULL){
		pch = strtok(line, ",\n");
		x = atof(pch);
		while (pch != NULL)
		{
			pch = strtok(NULL, ",\n");
			y = atof(pch);
			break;
		}
		Points.xCoordinates[counter] = x;
		Points.yCoordinates[counter] = y;
		Points.weights[counter] = 1.0f;
		Points.distances[counter] = 0.0f;

		counter++;
	}

	fclose(f);

	return Points;
}

void FreeSamplePoints(SamplePoints* Points){
	free(Points->xCoordinates);
	Points->xCoordinates = NULL;

	free(Points->yCoordinates);
	Points->yCoordinates = NULL;

	free(Points->weights);
	Points->weights = NULL;

	// By Guiming @ 2016-09-01
	free(Points->distances);
	Points->distances = NULL;
}

// write to .csv file
void WriteSamplePoints(SamplePoints* Points, const char * csvFile){
	FILE *f = fopen(csvFile, "w");
	if (f == NULL)
	{
		printf("Error opening file!\n");
		exit(1);
	}

	fprintf(f, "x, y\n");
	for (int p = 0; p < Points->numberOfPoints; p++){
		fprintf(f, "%f, %f\n", Points->xCoordinates[p], Points->yCoordinates[p]);
	}
	fclose(f);
}

// write to .csv file
void WriteSamplePoints(SamplePoints* Points, float* Hs, float* Ws, const char * csvFile){
	FILE *f = fopen(csvFile, "w");
	if (f == NULL)
	{
		printf("Error opening file!\n");
		exit(1);
	}

	fprintf(f, "x, y, h, w\n");
	for (int p = 0; p < Points->numberOfPoints; p++){
		fprintf(f, "%f, %f, %f, %f\n", Points->xCoordinates[p], Points->yCoordinates[p], Hs[p], Ws[p]);
	}
	fclose(f);
}

// this is a mask
AsciiRaster AllocateAsciiRaster(int nCols, int nRows, float xLLCorner, float yLLCorner, float cellSize, float noDataValue){
	AsciiRaster Ascii;

	Ascii.nCols = nCols;
	Ascii.nRows = nRows;
	Ascii.xLLCorner = xLLCorner;
	Ascii.yLLCorner = yLLCorner;
	Ascii.cellSize = cellSize;
	Ascii.noDataValue = noDataValue;

	int size = Ascii.nCols * Ascii.nRows;
	Ascii.elements = (float*)malloc(size*sizeof(float));

	for (int row = 0; row < Ascii.nRows; row++){
		for (int col = 0; col < Ascii.nCols; col++){
			//if (row < 2 || col < 2)
			//	Ascii.elements[row * nCols + col] = Ascii.noDataValue;
			//else
				Ascii.elements[row * nCols + col] = 0.0f;
		}
	}

	// By Guiming @ 2016-09-01
	//MarkBoundary(Ascii);

	return Ascii;
}

// copy a ascii raster
AsciiRaster CopyAsciiRaster(const AsciiRaster anotherAscii){
	AsciiRaster Ascii;

	Ascii.nCols = anotherAscii.nCols;
	Ascii.nRows = anotherAscii.nRows;
	Ascii.xLLCorner = anotherAscii.xLLCorner;
	Ascii.yLLCorner = anotherAscii.yLLCorner;
	Ascii.cellSize = anotherAscii.cellSize;
	Ascii.noDataValue = anotherAscii.noDataValue;

	int size = Ascii.nCols * Ascii.nRows;
	Ascii.elements = (float*)malloc(size*sizeof(float));

	for (int row = 0; row < Ascii.nRows; row++){
		for (int col = 0; col < Ascii.nCols; col++){
			Ascii.elements[row * Ascii.nCols + col] = anotherAscii.elements[row * Ascii.nCols + col];
		}
	}

	return Ascii;
}

// ascii raster read from a .asc file
AsciiRaster ReadAsciiRaster(char * asciiFile){
	FILE *f = fopen(asciiFile, "rt");
	if (f == NULL)
	{
		printf("Error opening file!\n");
		exit(1);
	}

	AsciiRaster Ascii;

	const int HEADER_LINE_LENGTH = 64;
	char hdrLine[HEADER_LINE_LENGTH];
	char* pch;
	float meta[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

	// read headers
	for (int i = 0; i < 6; i++){
		fgets(hdrLine, HEADER_LINE_LENGTH, f);
		pch = strtok(hdrLine, " \n");
		while (pch != NULL)
		{
			pch = strtok(NULL, "\n");
			meta[i] = atof(pch);
			break;
		}
	}

	Ascii.nCols = (int)meta[0];
	Ascii.nRows = (int)meta[1];
	Ascii.xLLCorner = meta[2];
	Ascii.yLLCorner = meta[3];
	Ascii.cellSize = meta[4];
	Ascii.noDataValue = meta[5];
	Ascii.elements = (float*)malloc(Ascii.nRows*Ascii.nCols*sizeof(float));

	const int DATA_LINE_LENGTH = Ascii.nCols * 32;
	char* datLine = (char*)malloc(DATA_LINE_LENGTH*sizeof(char));

	int row_counter = 0;
	while (fgets(datLine, DATA_LINE_LENGTH, f) != NULL){
		int col_counter = 0;
		pch = strtok(datLine, " \n");
		Ascii.elements[row_counter*Ascii.nCols+col_counter] = atof(pch);
		while (pch != NULL)
		{
			pch = strtok(NULL, " ");
			if (pch != NULL && col_counter < Ascii.nCols - 1){
				col_counter++;
				Ascii.elements[row_counter*Ascii.nCols + col_counter] = atof(pch);
			}
		}
		row_counter++;
	}
	free(datLine);

	fclose(f);

	// By Guiming @ 2016-09-01
	// MarkBoundary(Ascii);

	return Ascii;
}

void FreeAsciiRaster(AsciiRaster* Ascii){
	free(Ascii->elements);
	Ascii->elements = NULL;
}

// write to .asc file
void WriteAsciiRaster(AsciiRaster* Ascii, const char * asciiFile){
	FILE *f = fopen(asciiFile, "w");
	if (f == NULL)
	{
		printf("Error opening file!\n");
		exit(1);
	}

	fprintf(f, "ncols %d\n", Ascii->nCols);
	fprintf(f, "nrows %d\n", Ascii->nRows);
	fprintf(f, "xllcorner %f\n", Ascii->xLLCorner);
	fprintf(f, "yllcorner %f\n", Ascii->yLLCorner);
	fprintf(f, "cellsize %f\n", Ascii->cellSize);
	fprintf(f, "NODATA_value %.0f\n", Ascii->noDataValue);

	for (int row = 0; row < Ascii->nRows; row++){
		for (int col = 0; col < Ascii->nCols; col++){
			fprintf(f, "%.16f ", Ascii->elements[row*Ascii->nCols+col]);
		}
		fprintf(f, "\n");
	}
	fclose(f);
}

// edge effects correction weights at each point, weights is allocated somewhere else
void EdgeCorrectionWeightsExact(SamplePoints Points, float *hs, AsciiRaster Ascii, float *weights, int nThreads){
	float cellArea = Ascii.cellSize * Ascii.cellSize;
	float p_x, p_y, cell_x, cell_y;
	float ew;
	float h2;
#pragma omp parallel for schedule(OMP_SCHEDULE_PNTS) num_threads(nThreads) private(p_x, p_y, cell_x, cell_y, ew, h2)
	for (int p = 0; p < Points.numberOfPoints; p++){
		//printf("%6d / %6d\n", p, Points.numberOfPoints);
		h2 = hs[p] * hs[p];

		// By Guiming @ 2016-09-01
		if(Points.distances[p] >= CUT_OFF_FACTOR * h2){ // pnts too far away from the study area boundary, skip to save labor!
			weights[p] = 1.0f;
			//printf("bypassed! %f %f %d\n", Points.distances[p], 9.0 * h2, nThreads);
		}
		else{
			p_x = Points.xCoordinates[p];
			p_y = Points.yCoordinates[p];
			ew = 0.0f;

			// added by Guiming @2016-09-11
			// narrow down the row/col range
			int row_lower = 0;
			int row_upper = Ascii.nRows - 1;
			int col_lower = 0;
			int col_upper = Ascii.nCols - 1;

			if(NARROW){
				int r = YCOORD_TO_ROW(p_y + SQRT_CUT_OFF_FACTOR * hs[p], Ascii.nRows, Ascii.yLLCorner, Ascii.cellSize);
				row_lower = MAX(0, r);
				row_upper = MIN(Ascii.nRows - 1, YCOORD_TO_ROW(p_y - SQRT_CUT_OFF_FACTOR * hs[p], Ascii.nRows, Ascii.yLLCorner, Ascii.cellSize));
				col_lower = MAX(0, XCOORD_TO_COL(p_x - SQRT_CUT_OFF_FACTOR * hs[p], Ascii.xLLCorner, Ascii.cellSize));
				col_upper = MIN(Ascii.nCols - 1, XCOORD_TO_COL(p_x + SQRT_CUT_OFF_FACTOR * hs[p], Ascii.xLLCorner, Ascii.cellSize));
			}

			for (int row = row_lower; row <= row_upper; row++){
				for (int col = col_lower; col <= col_upper; col++){
					if (Ascii.elements[row*Ascii.nCols+col] != Ascii.noDataValue){
						cell_x = COL_TO_XCOORD(col, Ascii.xLLCorner, Ascii.cellSize);
						cell_y = ROW_TO_YCOORD(row, Ascii.nRows, Ascii.yLLCorner, Ascii.cellSize);
						float d2 = Distance2(p_x, p_y, cell_x, cell_y);
						ew += GaussianKernel(h2, d2) * cellArea;
					}
				}
			}
			weights[p] = 1.0 / ew;
		}
		if(BRUNSDON){
			weights[p] = 1.0; // Brunsdon 1995 parameter
		}
	}
}

void EdgeCorrectionWeightsExact(SamplePoints Points, float h, AsciiRaster Ascii, float *weights, int nThreads){
	float cellArea = Ascii.cellSize * Ascii.cellSize;
	float p_x, p_y, cell_x, cell_y;
	float ew;
	float h2 = h * h;
#pragma omp parallel for schedule(OMP_SCHEDULE_PNTS) num_threads(nThreads) private(p_x, p_y, cell_x, cell_y, ew) firstprivate(h2)
	for (int p = 0; p < Points.numberOfPoints; p++){
		//printf("%6d / %6d\n", p, Points.numberOfPoints);

		// By Guiming @ 2016-09-01
		if(Points.distances[p] >= CUT_OFF_FACTOR * h2){ // pnts too far away from the study area boundary, skip to save labor!
			weights[p] = 1.0f;
			//printf("bypassed! %f %f %d\n", Points.distances[p], 9.0 * h2, nThreads);
		}
		else{
			p_x = Points.xCoordinates[p];
			p_y = Points.yCoordinates[p];
			ew = 0.0f;

			// added by Guiming @2016-09-11
			// narrow down the row/col range
			int row_lower = 0;
			int row_upper = Ascii.nRows - 1;
			int col_lower = 0;
			int col_upper = Ascii.nCols - 1;
			if(NARROW){
				int r = YCOORD_TO_ROW(p_y + SQRT_CUT_OFF_FACTOR * h, Ascii.nRows, Ascii.yLLCorner, Ascii.cellSize);
				row_lower = MAX(0, r);
				row_upper = MIN(Ascii.nRows - 1, YCOORD_TO_ROW(p_y - SQRT_CUT_OFF_FACTOR * h, Ascii.nRows, Ascii.yLLCorner, Ascii.cellSize));
				col_lower = MAX(0, XCOORD_TO_COL(p_x - SQRT_CUT_OFF_FACTOR * h, Ascii.xLLCorner, Ascii.cellSize));
				col_upper = MIN(Ascii.nCols - 1, XCOORD_TO_COL(p_x + SQRT_CUT_OFF_FACTOR * h, Ascii.xLLCorner, Ascii.cellSize));
			}

			for (int row = row_lower; row <= row_upper; row++){
				for (int col = col_lower; col <= col_upper; col++){
					if (Ascii.elements[row*Ascii.nCols+col] != Ascii.noDataValue){
						cell_x = COL_TO_XCOORD(col, Ascii.xLLCorner, Ascii.cellSize);
						cell_y = ROW_TO_YCOORD(row, Ascii.nRows, Ascii.yLLCorner, Ascii.cellSize);
						float d2 = Distance2(p_x, p_y, cell_x, cell_y);
						ew += GaussianKernel(h2, d2) * cellArea;
					}
				}
			}
			weights[p] = 1.0 / ew;
		}
		if(BRUNSDON){
			weights[p] = 1.0; // Brunsdon 1995 parameter
		}
	}
}


float* AllocateEdgeCorrectionWeights(SamplePoints Points){
	return (float*)malloc(Points.numberOfPoints*sizeof(float));
}

void FreeEdgeCorrectionWeights(float* weights){
	free(weights);
	weights = NULL;
}

float* AllocateBandWidths(SamplePoints Points){
	return (float*)malloc(Points.numberOfPoints*sizeof(float));
}

void FreeBandWidths(float* hs){
	free(hs);
	hs = NULL;
}

float* AllocateDen(SamplePoints Points){ // n is number of points
	return (float*)malloc(Points.numberOfPoints*sizeof(float));
}
void FreeDen(float* den){
	free(den);
	den = NULL;
}

// check whether the result from sequential computation and that from parallel computation agree
void CheckResults(AsciiRaster AsciiSEQ, AsciiRaster AsciiPARA){
	float eps = 0.000001f;

	int n = AsciiSEQ.nCols * AsciiSEQ.nRows;

	for(int i = 0; i < n; i++){
		if(abs(AsciiSEQ.elements[i] - AsciiPARA.elements[i]) > eps){
			printf("TEST FAILED. Result from parallel computation does not match that from sequential computation.\n");
			return;
		}
	}
	printf("TEST PASSED. Result from parallel computation does match that from sequential computation.\n");
}

// compute the optimal Maximum Likelihood Estimation fixed bandwidth
// By Guiming @ 2016-02-26
float MLE_FixedBandWidth(AsciiRaster Ascii, SamplePoints Points, float *edgeWeights, float h, int nThreads){

	//float *hs = AllocateBandWidths(Points);

	float hA = h/10;
	float hD = 4 * h;

	if(BRUNSDON){
		hA = 0.05;
		hD = 0.07;
	}

	float width = hD - hA;
	float epsilon = width/100;
	float factor = 1 + sqrtf(5.0f);
	int iteration = 0;

	bool condition = width > epsilon;
	if(BRUNSDON){
		condition = iteration < 15;
	}

	while(condition){
		if(DEBUG){
			printf("iteration: %d ", iteration);
			printf("hD: %.6f ", hD);
			printf("hA: %.6f ", hA);
		}

		float hB = hA + width / factor;
		float hC = hD - width / factor;
		float LoghB = LogLikelihood(Ascii, Points, edgeWeights, hB, true, nThreads);
		float LoghC = LogLikelihood(Ascii, Points, edgeWeights, hC, true, nThreads);

		if(LoghB > LoghC){
			hD = hC;
			if(DEBUG) printf("LoghB: %.6f \n", LoghB);
		}
		else{
			hA = hB;
			if(DEBUG) printf("LoghC: %.6f \n", LoghC);
		}

		width = hD - hA;

		iteration += 1;

		condition = width > epsilon;
		if(BRUNSDON){
			condition = iteration < 15;
		}

	}
	return (hA + hD) / 2;
}

// the log likelihood given bandwidth h
// By Guiming @ 2016-02-26
float LogLikelihood(AsciiRaster Ascii, SamplePoints Points, float *edgeWeights, float h, bool update, int nThreads){
	int numPoints = Points.numberOfPoints;
	float logL = 0.0f; // log likelihood

	// update edge correction weights
	if(UPDATEWEIGHTS && update){
		EdgeCorrectionWeightsExact(Points, h, Ascii, edgeWeights, nThreads);
	}

	#pragma omp parallel for schedule(OMP_SCHEDULE_PNTS) num_threads(nThreads) reduction(+: logL)
	for(int i = 0; i < numPoints; i++){
		float pi_x = Points.xCoordinates[i];
		float pi_y = Points.yCoordinates[i];

		float den = 0.0f;
		for (int j = 0; j < numPoints; j++){
			if(j != i){
				float pj_x = Points.xCoordinates[j];
				float pj_y = Points.yCoordinates[j];
				float pj_w = Points.weights[j];
				float pj_ew = edgeWeights[j];

				float d2 = Distance2(pi_x, pi_y, pj_x, pj_y);
				den += GaussianKernel(h * h, d2) * pj_w *pj_ew;
			}
		}

		logL += logf(den + EPSILONDENSITY);

		if(BRUNSDON){
			logL += log(den/(numPoints - 1) + EPSILONDENSITY);
		}
	}

	return logL;
}

// the log likelihood given bandwidth h
// By Guiming @ 2016-02-26
// float* den0 : density based on all points, including itself
// float* den1 : leave one out density
float LogLikelihoodAdaptive(AsciiRaster Ascii, SamplePoints Points, float *edgeWeights, float* hs, bool update, float* den0, float* den1, int nThreads){
	int numPoints = Points.numberOfPoints;
	float logL = 0.0; // log likelihood

	if(UPDATEWEIGHTS && update){
		EdgeCorrectionWeightsExact(Points, hs, Ascii, edgeWeights, nThreads);
	}

	#pragma omp parallel for schedule(OMP_SCHEDULE_PNTS) num_threads(nThreads) reduction(+:logL)
	for(int i = 0; i < numPoints; i++){
		float pi_x = Points.xCoordinates[i];
		float pi_y = Points.yCoordinates[i];

		float den = 0.0f;
		float den_itself = 0.0f;

		for (int j = 0; j < numPoints; j++){
			float pj_x = Points.xCoordinates[j];
			float pj_y = Points.yCoordinates[j];
			float pj_h = hs[j];
			float pj_w = Points.weights[j];
			float pj_ew = edgeWeights[j];

			float d2 = Distance2(pi_x, pi_y, pj_x, pj_y);

			if(j == i){
				den_itself += GaussianKernel(pj_h * pj_h, d2) * pj_w *pj_ew; // / numPoints;
			}
			else{
				den += GaussianKernel(pj_h * pj_h, d2) * pj_w *pj_ew;
			}
		}


		logL = logL + logf(den + EPSILONDENSITY);
		//printf("den: %.6f log(den): %.6f\n", den, log(den));
		if(BRUNSDON){
			logL = logL + log(den/(numPoints  - 1)+ EPSILONDENSITY);
		}

		if(den0 != NULL){
			den0[i] = den + den_itself;
			if(BRUNSDON){
				den0[i] = (den + den_itself)/numPoints;
			}
		}
		if(den1 != NULL){
			den1[i] = den;
			if(BRUNSDON){
				den1[i] = den / (numPoints - 1);
			}
		}
	}

	return logL;
}

// compute the log likelihood given a center (h0, alpha0) and step (stepH, stepA)
// By Guiming @ 2016-03-06
/*
 return 9 elements log likelihood in float* logLs
**/
void hj_likelihood(AsciiRaster Ascii, SamplePoints Points, float *edgeWeights, float h0, float alpha0, float stepH, float stepA, int lastdmax, float* logLs, int nThreads){
    int n = Points.numberOfPoints;
    float* den0 = AllocateDen(Points);
    float* hs = AllocateBandWidths(Points);

    float gml;

    // the center (h0, alpha0)
    if(lastdmax == -1){ // avoid unnecessary [expensive] computation
	    for(int i = 0; i < n; i++){
	    	hs[i] = h0;
	    }
	    LogLikelihoodAdaptive(Ascii, Points, edgeWeights, hs, true, den0, NULL, nThreads);
	    gml = compGML(den0, n);
	    for(int i = 0; i < n; i++){
	    	hs[i] = hs[i] * powf((den0[i] / gml), alpha0);
	    }
	    float L0 = LogLikelihoodAdaptive(Ascii, Points, edgeWeights, hs, true, NULL, NULL, nThreads);
	    logLs[0] = L0;
	}

    // (h0 - stepH, alpha0)
    if(lastdmax != 2){ // avoid unnecessary [expensive] computation
	    for(int i = 0; i < n; i++){
	    	hs[i] = h0 - stepH;
	    }
	    LogLikelihoodAdaptive(Ascii, Points, edgeWeights, hs, true, den0, NULL, nThreads);
	    gml = compGML(den0, n);
	    for(int i = 0; i < n; i++){
	    	hs[i] = hs[i] * powf((den0[i] / gml), alpha0);
	    }
	    float L1 = LogLikelihoodAdaptive(Ascii, Points, edgeWeights, hs, true, NULL, NULL, nThreads);
	    logLs[1] = L1;
	}

    // (h0 + stepH, alpha0)
    if(lastdmax != 1){
	    for(int i = 0; i < n; i++){
	    	hs[i] = h0 + stepH;
	    }
	    LogLikelihoodAdaptive(Ascii, Points, edgeWeights, hs, true, den0, NULL, nThreads);
	    gml = compGML(den0, n);
	    for(int i = 0; i < n; i++){
	    	hs[i] = hs[i] * powf((den0[i] / gml), alpha0);
	    }
	    float L2 = LogLikelihoodAdaptive(Ascii, Points, edgeWeights, hs, true, NULL, NULL, nThreads);
	    logLs[2] = L2;
	}

    // (h0, alpha0 + stepA)
    if(lastdmax != 4){
	    for(int i = 0; i < n; i++){
	    	hs[i] = h0;
	    }
	    LogLikelihoodAdaptive(Ascii, Points, edgeWeights, hs, true, den0, NULL, nThreads);
	    gml = compGML(den0, n);
	    for(int i = 0; i < n; i++){
	    	hs[i] = hs[i] * powf((den0[i] / gml), alpha0 + stepA);
	    }
	    float L3 = LogLikelihoodAdaptive(Ascii, Points, edgeWeights, hs, true, NULL, NULL, nThreads);
	    logLs[3] = L3;
	}

    // (h0, alpha0 - stepA)
    if(lastdmax != 3){
	    for(int i = 0; i < n; i++){
	    	hs[i] = h0;
	    }
	    LogLikelihoodAdaptive(Ascii, Points, edgeWeights, hs, true, den0, NULL, nThreads);
	    gml = compGML(den0, n);
	    for(int i = 0; i < n; i++){
	    	hs[i] = hs[i] * powf((den0[i] / gml), alpha0 - stepA);
	    }
	    float L4 = LogLikelihoodAdaptive(Ascii, Points, edgeWeights, hs, true, NULL, NULL, nThreads);
	    logLs[4] = L4;
	}

    FreeDen(den0);
    FreeBandWidths(hs);
}

// compute the optimal h and alpha (parameters for calculating the optimal adaptive bandwith)
// By Guiming @ 2016-03-06
/*
 return 3 optmal parameters in float* optParas (optH, optAlpha, LogLmax)
**/
void hooke_jeeves(AsciiRaster Ascii, SamplePoints Points, float *edgeWeights, float h0, float alpha0, float stepH, float stepA, float* optParas, int nThreads){
	float* Ls = (float*)malloc(5 * sizeof(float));
	hj_likelihood(Ascii, Points, edgeWeights, h0, alpha0, stepH, stepA, -1, Ls, nThreads);

	float Lmax = Ls[0];

	float s = stepH / 20;
	float a = stepA / 20;

	// Brunsdon 1995 parameter
	if(BRUNSDON){
		s = 0.00001f;
		a = 0.0001f;
	}

	int iteration = 0;
    while ((stepH > s || stepA > a) &&  iteration <= MAX_NUM_ITERATIONS){

        float Lmax0 = Lmax;
        int dmax = 0;
        for(int i = 0; i < 5; i++){
            if(Ls[i] > Lmax){
            	Lmax = Ls[i];
                dmax = i;
            }
        }

        if(DEBUG)
        	printf ("iteration: %d center: (%.5f %.5f) steps: (%.5f %.5f) dmax: %d Lmax: %.5f \n", iteration, h0, alpha0, stepH, stepA, dmax, Lmax);

        if(dmax == 0){
            stepH = stepH / 2;
            stepA = stepA / 2;
        }

        else{
            if(dmax == 1){
                h0 = h0 - stepH;
                alpha0 = alpha0;
                Ls[2] = Ls[0]; // avoid unnecessary [expensive] computation
                Ls[0] = Ls[1];
            }
            if(dmax == 2){
                h0 = h0 + stepH;
                alpha0 = alpha0;
                Ls[1] = Ls[0];
                Ls[0] = Ls[2];
            }
            if (dmax == 3){
                h0 = h0;
                alpha0 = alpha0 + stepA;
                Ls[3] = Ls[0];
                Ls[0] = Ls[4];
            }
            if(dmax == 4){
                h0 = h0;
                alpha0 = alpha0 - stepA;
                Ls[3] = Ls[0];
                Ls[0] = Ls[4];
            }
        }
	    hj_likelihood(Ascii, Points, edgeWeights, h0, alpha0, stepH, stepA, dmax, Ls, nThreads);

	    iteration++;
    }

    optParas[0] = h0;
    optParas[1] = alpha0;

    // Brunsdon 1995 parameter
    /*optParas[0] = 0.0478;
    optParas[1] = -1.48f;*/

    optParas[2] = Lmax;

    free(Ls);
    Ls = NULL;
}

float compGML(float* den0, int numPoints){
	float gml = 0.0f;
	for(int i = 0; i < numPoints; i++){
		gml = gml + logf(den0[i]);
	}
	gml = expf(gml / numPoints);
	return gml;
}

// mark the boundary cells on a raster representing the study area
// By Guiming @ 2016-09-01, SIMPLIFIED VERSION
void MarkBoundary(AsciiRaster Ascii, int nThreads){
	// now we mark the cells on the study area boundary as 1 and those inside as 0
#pragma omp parallel for schedule(OMP_SCHEDULE_CELLS) num_threads(nThreads)
	for(int row = 0; row < Ascii.nRows; row++){
		for(int col = 0; col < Ascii.nCols; col++){

			if(Ascii.elements[row * Ascii.nCols + col] == Ascii.noDataValue)
				continue;

			if(row == 0 || (row == Ascii.nRows - 1) || col == 0 || (col == Ascii.nCols - 1)){
					Ascii.elements[row * Ascii.nCols + col] = 1.0f;
					continue;
			}

			if(Ascii.elements[(row - 1) * Ascii.nCols + col - 1] == Ascii.noDataValue){
				Ascii.elements[row * Ascii.nCols + col] = 1.0f;
				continue;
			}
			if(Ascii.elements[row * Ascii.nCols + col - 1] == Ascii.noDataValue){
				Ascii.elements[row * Ascii.nCols + col] = 1.0f;
				continue;
			}
			if(Ascii.elements[(row + 1) * Ascii.nCols + col - 1] == Ascii.noDataValue){
				Ascii.elements[row * Ascii.nCols + col] = 1.0f;
				continue;
			}

			if(Ascii.elements[(row - 1) * Ascii.nCols + col] == Ascii.noDataValue){
				Ascii.elements[row * Ascii.nCols + col] = 1.0f;
				continue;
			}
			if(Ascii.elements[(row + 1) * Ascii.nCols + col] == Ascii.noDataValue){
				Ascii.elements[row * Ascii.nCols + col] = 1.0f;
				continue;
			}

			if(Ascii.elements[(row - 1) * Ascii.nCols + col + 1] == Ascii.noDataValue){
				Ascii.elements[row * Ascii.nCols + col] = 1.0f;
				continue;
			}
			if(Ascii.elements[row * Ascii.nCols + col + 1] == Ascii.noDataValue){
				Ascii.elements[row * Ascii.nCols + col] = 1.0f;
				continue;
			}
			if(Ascii.elements[(row + 1) * Ascii.nCols + col + 1] == Ascii.noDataValue){
				Ascii.elements[row * Ascii.nCols + col] = 1.0f;
				continue;
			}

			Ascii.elements[row * Ascii.nCols + col] = 0.0f;
		}
	}
}

// compute the closest distances from sample points to study area boundary
// By Guiming @ 2016-09-01
void CalcDist2Boundary(SamplePoints Points, AsciiRaster Ascii, int nThreads){

	// mark the boundary first
	//MarkBoundary(Ascii, nThreads);

	float p_x, p_y, cell_x, cell_y;

#pragma omp parallel for schedule(static) num_threads(nThreads) private(p_x, p_y, cell_x, cell_y)
	for (int p = 0; p < Points.numberOfPoints; p++){
		float minDist = FLT_MAX;
		p_x = Points.xCoordinates[p];
		p_y = Points.yCoordinates[p];

		for (int row = 0; row < Ascii.nRows; row++){
			for (int col = 0; col < Ascii.nCols; col++){
				if (Ascii.elements[row*Ascii.nCols+col] == 1.0f){ // cells on boundary
					cell_x = COL_TO_XCOORD(col, Ascii.xLLCorner, Ascii.cellSize);
					cell_y = ROW_TO_YCOORD(row, Ascii.nRows, Ascii.yLLCorner, Ascii.cellSize);
					float d2 = Distance2(p_x, p_y, cell_x, cell_y);

					if(d2 < minDist){
						minDist = d2;
					}
				}
			}
		}

		Points.distances[p] = minDist;
		//printf("p: %d Points.distances[p]: %f minDist: %f\n", p, Points.distances[p]);
	}
}

// By Guiming @ 2016-09-04
SamplePoints CopySamplePoints(const SamplePoints anotherPoints){ // copy points
	int n = anotherPoints.numberOfPoints;
	SamplePoints Points = AllocateSamplePoints(n);
	Points.numberOfPoints = n;
	for(int p = 0; p < n; p++){
		Points.xCoordinates[p] = anotherPoints.xCoordinates[p];
		Points.yCoordinates[p] = anotherPoints.yCoordinates[p];
		Points.weights[p] = anotherPoints.weights[p];
		Points.distances[p] = anotherPoints.distances[p];
	}
	return Points;
}

// comparison function for sort
// By Guiming @ 2016-09-04
int compare ( const void *pa, const void *pb )
{
    const float *a = (const float *)pa;
    const float *b = (const float *)pb;
    if(a[0] == b[0])
        return a[1] - b[1];
    else
        return a[0] > b[0] ? 1 : -1;
}

// sort the sample points on their distances to study area boundary
// By Guiming @ 2016-09-04
void SortSamplePoints(SamplePoints Points){
	int n = Points.numberOfPoints;

	SamplePoints temPoints = CopySamplePoints(Points);

	float distances[n][2];
  for (int i = 0; i < n; i++)
	{
      distances[i][0] = Points.distances[i];
      distances[i][1] = i * 1.0f;
  }

  qsort(distances, n, sizeof(distances[0]), compare);

	for (int i = 0; i < n; i++)
	{
		int idx = (int)distances[i][1];
		Points.xCoordinates[i] = temPoints.xCoordinates[idx];
		Points.yCoordinates[i] = temPoints.yCoordinates[idx];
		Points.weights[i] = temPoints.weights[idx];
		Points.distances[i] = temPoints.distances[idx];
	}
	FreeSamplePoints(&temPoints);

}
