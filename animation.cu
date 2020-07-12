#include <gd.h>
#include <stdio.h>

#include <fenv.h>
#include <math.h>
#include <errno.h>

// ------------------------
// TO BE CUSTOMIZED BY USER
// ------------------------

// RENDERING PARAMETERS
#define sharpness     2000                                          // number of pixels specifying PNG pngWidth
#define maxIter       500                                          // set higher for highly zoomed-in pictures
#define nFrames       900
#define PI            3.14159265

// ------------------------
// COMPLEX DOMAIN
double centerRe = 0;
double centerIm = 0;

double epsilon = 5;

double radius = 0.7885;

// See the bottom of this code for a discussion of some output possibilities.
char*   filenameF =   "output/JuliaSet%05d.png";
void create_frame(int iteration);

__global__
void fillColor(int n, int H, int W, int* color, int* grey, int blue, double reStart, double reEnd, double imStart, double imEnd, double radius, double a) {

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
    color[T] = blue;
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

  int shade = 255 - ((numberOfIterations * 255) / maxIter);
  color[T] = grey[shade];
}

int main(){
  for (int i = 0; i < nFrames; i++) {
    create_frame(i);
  }
}

void create_frame(int frame) {
  FILE*       outfile;                               // defined in stdio
  gdImagePtr  image;                                 // a GD image object
  char        filename[80];
  int         i, T, x, y;                            // array subscripts
  int         blue, grey[256];       // red, all possible shades of grey
  int*        d_grey;


  double reStart = centerRe - epsilon;
  double reEnd = centerRe + epsilon;
  double imStart = centerIm - epsilon;
  double imEnd = centerIm + epsilon;

  double a = 2 * PI * frame / nFrames;

  int pngWidth = sharpness;
  int pngHeight = pngWidth * (imEnd - imStart) / (reEnd - reStart);
  int N = pngWidth * pngHeight;

  //int** color = make2DintArray(pngWidth, pngHeight);
  int* color = (int*) malloc(N*sizeof(int));
  int* d_color;

  printf("width: %i\n", pngWidth);
  printf("height: %i\n", pngHeight);

  image = gdImageCreate(pngWidth, pngHeight);

  blue  = gdImageColorAllocate(image, 0, 0, 255);
  
  for (i=0; i<256; i++){
    grey[i] = gdImageColorAllocate(image, i,i,i);
  }

  //void fillColor(int n, int H, int W, int* color, int* grey, int blue) {
  cudaMalloc(&d_grey, 256*sizeof(int)); 
  cudaMalloc(&d_color, N*sizeof(int));

  cudaMemcpy(d_grey, grey, 256*sizeof(int), cudaMemcpyHostToDevice);

  // Calculate power tower convergence / divergence
  fillColor<<<(pngWidth*pngHeight+255)/256, 256>>>(N, pngHeight, pngWidth, d_color, d_grey, blue, reStart, reEnd, imStart, imEnd, radius, a);
  cudaMemcpy(color, d_color, N*sizeof(int), cudaMemcpyDeviceToHost);

  for (T=0; T<pngWidth*pngHeight; T++) {
    x = T % pngHeight;
    y = T / pngHeight;
    gdImageSetPixel(image, x, y, color[T]);
  }

  // Free 2D array
  free(color);
  cudaFree(d_color);
  cudaFree(d_grey);
  // Finally, write the image out to a file.
  sprintf(filename, filenameF, frame);
  printf("Creating output file '%s'.\n", filename);
  outfile = fopen(filename, "wb");
  gdImagePng(image, outfile);
  fclose(outfile);
}