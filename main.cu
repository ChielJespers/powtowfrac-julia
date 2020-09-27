#include <gd.h>
#include <stdio.h>

#include <fenv.h>
#include <math.h>
#include <errno.h>

// ------------------------
// TO BE CUSTOMIZED BY USER
// ------------------------

// RENDERING PARAMETERS
#define sharpness     10000                                          // number of pixels specifying PNG pngWidth
#define maxIter       500                                         // set higher for highly zoomed-in pictures
#define nFrames       900
#define PI            3.14159265

// ------------------------
// COMPLEX DOMAIN
double centerRe = 0;
double centerIm = 0;

double epsilon = 5;

typedef struct RgbColor
{
  unsigned char r;
  unsigned char g;
  unsigned char b;
} RgbColor;

typedef struct HsvColor
{
  unsigned char h;
  unsigned char s;
  unsigned char v;
} HsvColor;

RgbColor HsvToRgb(HsvColor hsv)
{
  RgbColor rgb;
  unsigned char region, remainder, p, q, t;

  if (hsv.s == 0)
  {
    rgb.r = hsv.v;
    rgb.g = hsv.v;
    rgb.b = hsv.v;
    return rgb;
  }

  region = hsv.h / 43;
  remainder = (hsv.h - (region * 43)) * 6; 

  p = (hsv.v * (255 - hsv.s)) >> 8;
  q = (hsv.v * (255 - ((hsv.s * remainder) >> 8))) >> 8;
  t = (hsv.v * (255 - ((hsv.s * (255 - remainder)) >> 8))) >> 8;

  switch (region)
  {
    case 0:
      rgb.r = hsv.v; rgb.g = t; rgb.b = p;
      break;
    case 1:
      rgb.r = q; rgb.g = hsv.v; rgb.b = p;
      break;
    case 2:
      rgb.r = p; rgb.g = hsv.v; rgb.b = t;
      break;
    case 3:
      rgb.r = p; rgb.g = q; rgb.b = hsv.v;
      break;
    case 4:
      rgb.r = t; rgb.g = p; rgb.b = hsv.v;
      break;
    default:
      rgb.r = hsv.v; rgb.g = p; rgb.b = q;
      break;
  }

  return rgb;
}

HsvColor RgbToHsv(RgbColor rgb)
{
  HsvColor hsv;
  unsigned char rgbMin, rgbMax;

  rgbMin = rgb.r < rgb.g ? (rgb.r < rgb.b ? rgb.r : rgb.b) : (rgb.g < rgb.b ? rgb.g : rgb.b);
  rgbMax = rgb.r > rgb.g ? (rgb.r > rgb.b ? rgb.r : rgb.b) : (rgb.g > rgb.b ? rgb.g : rgb.b);

  hsv.v = rgbMax;
  if (hsv.v == 0)
  {
    hsv.h = 0;
    hsv.s = 0;
    return hsv;
  }

  hsv.s = 255 * long(rgbMax - rgbMin) / hsv.v;
  if (hsv.s == 0)
  {
    hsv.h = 0;
    return hsv;
  }

  if (rgbMax == rgb.r)
    hsv.h = 0 + 43 * (rgb.g - rgb.b) / (rgbMax - rgbMin);
  else if (rgbMax == rgb.g)
    hsv.h = 85 + 43 * (rgb.b - rgb.r) / (rgbMax - rgbMin);
  else
    hsv.h = 171 + 43 * (rgb.r - rgb.g) / (rgbMax - rgbMin);

  return hsv;
}

// See the bottom of this code for a discussion of some output possibilities.
char*   filenameF =   "output/JuliaSet%05d.png";
void create_frame(int iteration);

__global__
void fillColor(int n, int H, int W, int* color, int* palette, int black, double reStart, double reEnd, double imStart, double imEnd, double radius, double a) {

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
    color[T] = black;
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

  int shade = (255 * numberOfIterations) / maxIter;
  color[T] = numberOfIterations < maxIter ? palette[shade] : black;
}

int main(){
  create_frame(300);
}

void create_frame(int frame) {
  FILE*       outfile;                               // defined in stdio
  gdImagePtr  image;                                 // a GD image object
  char        filename[80];
  int         i, T, x, y;                            // array subscripts
  int         black, palette[256];       // red, all possible shades of palette
  int*        d_palette;
  HsvColor    col_hsv;
  RgbColor    col_rgb;

  double reStart = centerRe - epsilon;
  double reEnd = centerRe + epsilon;
  double imStart = centerIm - epsilon;
  double imEnd = centerIm + epsilon;

  double a = 2 * PI * frame / nFrames;
  double radius = 0.7885;

  printf("radius: %f\n", radius);

  int pngWidth = sharpness;
  int pngHeight = pngWidth * (imEnd - imStart) / (reEnd - reStart);
  int N = pngWidth * pngHeight;

  //int** color = make2DintArray(pngWidth, pngHeight);
  int* color = (int*) malloc(N*sizeof(int));
  int* d_color;

  printf("width: %i\n", pngWidth);
  printf("height: %i\n", pngHeight);

  image = gdImageCreate(pngWidth, pngHeight);

  black = gdImageColorAllocate(image, 0, 0, 0);
  
  for (i=0; i<255; i++){
    //hue = int(255 * m / MAX_ITER)
    //saturation = 255
    //value = 255 if m < MAX_ITER else 0

    col_hsv.h = i;
    col_hsv.s = 255;
    col_hsv.v = (i == 255 ? 0 : 255);
    col_rgb = HsvToRgb(col_hsv);
    palette[i] = gdImageColorAllocate(image, col_rgb.r, col_rgb.g, col_rgb.b);
  }

  palette[255] = gdImageColorAllocate(image, 0, 0, 0);

  //void fillColor(int n, int H, int W, int* color, int* palette, int black) {
  cudaMalloc(&d_palette, 256*sizeof(int)); 
  cudaMalloc(&d_color, N*sizeof(int));

  cudaMemcpy(d_palette, palette, 256*sizeof(int), cudaMemcpyHostToDevice);

  // Calculate power tower convergence / divergence
  fillColor<<<(pngWidth*pngHeight+255)/256, 256>>>(N, pngHeight, pngWidth, d_color, d_palette, black, reStart, reEnd, imStart, imEnd, radius, a);
  cudaMemcpy(color, d_color, N*sizeof(int), cudaMemcpyDeviceToHost);

  for (T=0; T<pngWidth*pngHeight; T++) {
    x = T % pngHeight;
    y = T / pngHeight;
    gdImageSetPixel(image, x, y, color[T]);
  }

  // Free 2D array
  free(color);
  cudaFree(d_color);
  cudaFree(d_palette);
  // Finally, write the image out to a file.
  sprintf(filename, filenameF, frame);
  printf("Creating output file '%s'.\n", filename);
  outfile = fopen(filename, "wb");
  gdImagePng(image, outfile);
  fclose(outfile);
}