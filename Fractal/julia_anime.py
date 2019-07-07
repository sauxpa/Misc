#!/usr/bin/env python
# coding: utf-8

# need to plot in a separate window to see the animation
#get_ipython().run_line_magic('matplotlib', 'qt')

import argparse, sys

import fractal
import numpy as np
from PIL import Image
from matplotlib import cm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
plt.style.use('dark_background')

matplotlib.use('TkAgg')  # or 'GTK3Cairo'

# Get arguments
parser=argparse.ArgumentParser()
parser.add_argument('--xmin', default=-2.0)
parser.add_argument('--xmax', default=2.0)
parser.add_argument('--ymin', default=-2.0)
parser.add_argument('--ymax', default=2.0)
parser.add_argument('--width', default=1920)
parser.add_argument('--height', default=1080)
parser.add_argument('--maxiter', default=256)
parser.add_argument('--interval', default=100, help='In milliseconds')
parser.add_argument('--method', default='parallel')
parser.add_argument('--c0', default=0.7885, help='Initial c in Julia iteration z -> z**2+c', type=float)
parser.add_argument('--filename', '-f', default='', help='File to save the animation')
parser.add_argument('--verbose', '-v', default=False)

# Set arguments
args = parser.parse_args()
xmin = args.xmin
xmax = args.xmax
ymin = args.ymin
ymax = args.ymax
width = args.width
height = args.height
frames = 5 # freeze otherwise
maxiter = args.maxiter
interval = args.interval
method = args.method
c0 = args.c0
filename = args.filename
verbose = args.verbose

julia = fractal.julia_set(xmin, xmax, ymin, ymax, width, height, maxiter, c=c0, method=method)

fig = plt.figure()
im = plt.imshow(julia, interpolation='none', animated=True)

def init():
    julia = fractal.julia_set(xmin, xmax, ymin, ymax, width, height, maxiter, c=c0, method=method)
    im.set_array(julia)
    return [im]

def animate(i):
    theta = 2*np.pi*i/frames
    c = c0*np.exp(theta*1j)
    if verbose:
        print('{}/{}'.format(i, frames))
    julia = fractal.julia_set(xmin, xmax, ymin, ymax, width, height, maxiter, c=c, method=method)
    im.set_array(julia)
    return [im]

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=frames, interval=interval, blit=True, repeat=False)
plt.axis('off')

if len(filename):
    if '.mp4' in filename:
        writer = animation.FFMpegWriter(fps=5, bitrate=-1)
        anim.save(filename, writer=writer)
    else:
        anim.save(filename, writer='imagemagick')
else:
    plt.show(fig)
