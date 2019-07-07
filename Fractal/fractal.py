#!/usr/bin/env python
# coding: utf-8

import numpy as np
from numba import jit, guvectorize, float64, float32, complex128, complex64, int32, uint8

# Mandelbrot set

@jit(int32(complex64, int32))
def mandelbrot_iter(z, maxiter):
    nreal = 0
    real = 0
    imag = 0
    for n in range(maxiter):
        nreal = real*real - imag*imag + z.real
        imag = 2*real*imag + z.imag
        real = nreal;
        if real * real + imag * imag > 4.0:
            return n
    return 0

@guvectorize([(complex64[:], int32[:], int32[:])], '(n),()->(n)', target='parallel')
def mandelbrot_numpy(z, maxit, output):
    maxiter = maxit[0]
    for i in range(z.shape[0]):
        output[i] = mandelbrot_iter(z[i],maxiter)

@guvectorize([(complex64[:], int32[:], int32[:])], '(n),(n)->(n)', target='cuda')
def mandelbrot_numpy_cuda(z, maxit, output):
    maxiter = maxit[0]
    for i in range(z.shape[0]):
        zreal = z[i].real
        zimag = z[i].imag
        real = zreal
        imag = zimag
        output[i] = 0
        for n in range(maxiter):
            real2 = real*real
            imag2 = imag*imag
            if real2 + imag2 > 4.0:
                output[i] = n
                break
            imag = 2*real*imag + zimag
            real = real2 - imag2 + zreal

def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, maxiter, method='parallel'):
    r1 = np.linspace(xmin, xmax, width, dtype=np.float32)
    r2 = np.linspace(ymin, ymax, height, dtype=np.float32)
    z = r1 + r2[:,None]*1j
    mandelbrot = np.empty(z.shape, int)
    maxit = np.ones(z.shape, int) * maxiter

    if method == 'parallel':
        mandelbrot = mandelbrot_numpy(z, maxiter)
    elif method == 'cuda':
        mandelbrot = mandelbrot_numpy_cuda(z, maxit)
    elif method == 'single':
        for i in range(width):
            for j in range(height):
                mandelbrot[j,i] = mandelbrot_iter(r1[i]+r2[j]*1j, maxiter)
    else:
        raise NameError('Unknown method: {}'.format(method))

    return mandelbrot

# Julia set

@jit(int32(complex64, complex64, int32))
def julia_iter(z, c, maxiter):
    creal = c.real
    cimag = c.imag

    zreal = z.real
    zimag = z.imag
    zreal2 = zreal*zreal
    zimag2 = zimag*zimag

    output = 0
    while zimag2 + zreal2 <= 4:
        zimag = 2*zreal*zimag + cimag
        zreal = zreal2 - zimag2 + creal
        zreal2 = zreal*zreal
        zimag2 = zimag*zimag
        output += 1

    return output

@guvectorize([(complex64[:], complex64[:], int32[:], int32[:])], '(n),(),()->(n)', target='parallel')
def julia_numpy(z, c, maxit, output):
    maxiter = maxit[0]
    c = c[0]
    for i in range(z.shape[0]):
        output[i] = julia_iter(z[i], c, maxiter)

@guvectorize([(complex64[:], complex64[:], int32[:], int32[:])], '(n),(n),(n)->(n)', target='cuda')
def julia_numpy_cuda(z, c, maxit, output):
    maxiter = maxit[0]
    c = c[0]
    for i in range(z.shape[0]):
        creal = c.real
        cimag = c.imag

        zreal = z[i].real
        zimag = z[i].imag
        zreal2 = zreal*zreal
        zimag2 = zimag*zimag

        output_i = 0
        while zimag2 + zreal2 <= 4:
            zimag = 2*zreal*zimag + cimag
            zreal = zreal2 - zimag2 + creal
            zreal2 = zreal*zreal
            zimag2 = zimag*zimag
            output_i += 1

        output[i] = output_i

def julia_set(xmin, xmax, ymin, ymax, width, height, maxiter, method='parallel', c=0+0j):
    r1 = np.linspace(xmin, xmax, width, dtype=np.float32)
    r2 = np.linspace(ymin, ymax, height, dtype=np.float32)
    z = r1 + r2[:,None]*1j
    julia = np.empty(z.shape, int)
    maxit = np.ones(z.shape, int) * maxiter
    c_grid = np.ones(z.shape, int) * c
    c_grid = c_grid.astype(np.dtype('complex64'))

    if method == 'parallel':
        julia = julia_numpy(z, c, maxiter)
    elif method == 'cuda':
        julia = julia_numpy_cuda(z, c_grid, maxit)
    elif method == 'single':
        for i in range(width):
            for j in range(height):
                julia[j,i] = julia_iter(r1[i]+r2[j]*1j, c, maxiter)
    else:
        raise NameError('Unknown method: {}'.format(method))

    return julia
