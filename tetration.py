from PIL import Image, ImageDraw
from collections import defaultdict
from math import floor, ceil, pi

import numpy as np
import ctypes
import time
import cmath
from ctypes import *
from numpy.ctypeslib import ndpointer

from concat import *

def get_cuda_tetration(sharpness):
    dll = ctypes.CDLL('./powtowfrac.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.create_frame
    func.argtypes = [c_int, c_double, c_double, c_int, c_double, POINTER(c_double)]
    func.restype = ndpointer(dtype=ctypes.c_double, shape=(sharpness * sharpness,))
    return func

def cuda_tetration(sharpness, a, radius, maxIter, epsilon, res, tetr):
    res_p = res.ctypes.data_as(POINTER(c_double))

    return tetr(sharpness, a, radius, maxIter, epsilon, res_p)

def linear_interpolation(color1, color2, t):
    return color1 * (1 - t) + color2 * t 

def tetr_execute(sA, sRadius, sMaxiter, sSharpness, epsilon, frame):
    start = time.time()

    # Input variables
    sharpness = int(sSharpness)

    re = 0
    im = 0

    maxIter = int(sMaxiter)
    a = float(sA)
    radius = float(sRadius)

    # Calculation of window size, number of pixels
    reStart = re - epsilon
    reEnd = re + epsilon
    imStart = im - epsilon
    imEnd = im + epsilon

    pngWidth = int(sharpness)
    pngHeight = int(pngWidth * (imEnd - imStart) / (reEnd - reStart))

    N = pngWidth * pngHeight

    # Execution
    __cuda_tetration = get_cuda_tetration(sharpness)
    res = np.zeros(N).astype('float64')
    res = cuda_tetration(sharpness, a, radius, maxIter, epsilon, res, __cuda_tetration)
    print(res)
    elapsed = time.time() - start
    start = time.time()
    print "Time elapsed:", elapsed

    # Put results in picture
    outfile = 'raster/frame{:05d}.png'.format(frame)
    print "Start creating image ", outfile
    pic = Image.new('HSV', (pngWidth, pngHeight), (0, 0, 0))
    draw = ImageDraw.Draw(pic)

    black = (0,0,0)
    palette = range(256)
    for i in palette:
        h = i
        s = 255
        v = 255 if i < 255 else 0
        palette[i] = (h,s,v)

    histogram = defaultdict(lambda: 0)

    it = res

    # Create histogram of colors
    for T in xrange(N):
        if it[T] < maxIter:
            histogram[int(it[T])] += 1

    total = sum(histogram.values())
    hues = []
    h = 0
    for i in range(maxIter):
        h += (255 * histogram[i])
        hues.append(h)
    hues.append(h)

    if total > 0:
        hues = [hue / total for hue in hues]

    # Draw to image
    for T in xrange(N):
        x = T % pngHeight
        y = T / pngHeight
        shade = int(linear_interpolation(hues[int(floor(it[T]))], hues[int(ceil(it[T]))], it[T] % 1))
        color = palette[shade] if it[T] < maxIter else black
        draw.point([x,y], color)

    pic.convert('RGB').save(outfile, 'PNG')

    elapsed = time.time() - start
    print "Time elapsed:", elapsed

# NB only use for odd n
def n_root(t, n):
    if (t < 0):
        return -n_root(-t, n)
    else:
        return t ** (1. / n)

nframes = 600
k = 3.

def raster(S, G, R):
    sharpness = int(S / G)
    for J in range(1, G + 1):
        for I in range(1, G + 1):
            LB = -R + (2 * R * (I - 1)) / G + 1j * (R - (2 * R * (J - 1)) / G)
            C = LB + R / G - 1j * R / G
            (radius, a) = cmath.polar(C)
            epsilon = R / G
            tetr_execute(str(a), str(radius), '100', str(sharpness), epsilon, G * (J - 1) + I)

    concat_all(G, sharpness)

def n_root_param(t):
    return .5 + (n_root(t - .5, k) / (2. * n_root(.5, k)))

def power_param(t):
    if t <= .5:
        return (2. * t) ** k / 2
    else:
        return ((2. * (t - 1)) ** k / 2) + 1

# for i in range(nframes):
#     if i < 220:
#         continue
#     t = float(i) / nframes
#     s = power_param(t)
#     print(s)
#     z = cmath.exp(cmath.exp(2 * pi * s * 1j - cmath.exp(2 * pi * s * 1j)))
#     (radius, a) = cmath.polar(z)
#     tetr_execute(str(a), str(radius), '1000', i)

raster(4000, 200, 2.5)