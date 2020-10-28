#include <gd.h>
#include <stdio.h>

#include <fenv.h>
#include <math.h>
#include <errno.h>

// ------------------------
// TO BE CUSTOMIZED BY USER
// ------------------------

// RENDERING PARAMETERS
#define nFrames       300
#define PI            3.14159265

// ------------------------
// COMPLEX DOMAIN
double centerRe = 0;
double centerIm = 0;

__device__
double slog(double in) {
  double result = 0;

  while (in > 1) {
    result += 1;
    in = log(in);
  }

  return result - 1 + in;
}

__global__
void fillColor(int n, int H, int W, double* color, double reStart, double reEnd, double imStart, double imEnd, double radius, double a, int maxIter) {

  int T = blockIdx.x*blockDim.x + threadIdx.x;
  if (T >= n) return;

  int x = T % H;
  int y = T / H;
  double re = reStart + ((double) x / W * (reEnd - reStart));
  double im = imEnd - ((double) y / H * (imEnd - imStart));

  double nextRe, nextIm, logRe, logIm, powerRe, powerIm;

  int toggleOverflow = 0;                                          
  int numberOfIterations = 0;                                      
  if (re == 0 && im == 0){
    color[T] = maxIter;
  }
  else {
    logRe = log(radius);
    logIm = a;

    nextRe = re;
    nextIm = im;
    while (numberOfIterations < maxIter && toggleOverflow == 0)
    {
        powerRe = (nextRe * logRe - nextIm * logIm);
        powerIm = (nextRe * logIm + nextIm * logRe);

        if (powerRe > 700) {
            toggleOverflow = 1;
        }

        nextRe = exp(powerRe) * cos(powerIm);
        nextIm = exp(powerRe) * sin(powerIm);
        
        numberOfIterations += 1;
    }
  }

  double it = numberOfIterations == maxIter ? maxIter : numberOfIterations + 1 - slog(powerRe);
  color[T] = it;
}

// def tetr_execute(sA, sRadius, sMaxiter):

extern "C" {
double *create_frame(int sharpness, double a, double radius, int maxIter, double epsilon, double *res) {
  double reStart = centerRe - epsilon;
  double reEnd = centerRe + epsilon;
  double imStart = centerIm - epsilon;
  double imEnd = centerIm + epsilon;

  printf("a: %f\n", a);
  printf("radius: %f\n", radius);

  int pngWidth = sharpness;
  int pngHeight = pngWidth * (imEnd - imStart) / (reEnd - reStart);
  int N = pngWidth * pngHeight;

  res = (double*) malloc(N*sizeof(double));
  double* d_color;

  printf("width: %i\n", pngWidth);
  printf("height: %i\n", pngHeight);

  cudaMalloc(&d_color, N*sizeof(double));

  // Calculate power tower convergence / divergence
  fillColor<<<(pngWidth*pngHeight+255)/256, 256>>>(N, pngHeight, pngWidth, d_color, reStart, reEnd, imStart, imEnd, radius, a, maxIter);
  cudaMemcpy(res, d_color, N*sizeof(double), cudaMemcpyDeviceToHost);

  // Free 2D array
  cudaFree(d_color);
  // Finally, write the image out to a file.
  return res;
}
}