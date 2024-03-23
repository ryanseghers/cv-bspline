# cv-bspline
C++ OpenCV Bezier and B-Spline Library

## Introduction
This has Bezier and B-spline curve and surface implementations in C++ using OpenCV.
So far this always uses a uniform grid, meaning grid cells are square and 1x1.
If the B-spline control points are on a uniform grid, then the Bezier points also end up on a uniform grid.

## Segments (1D) and Cells (2D) vs Data Structures
In a sequence of segments, nSegments = nPoints - 1
Similarly, in a grid of cells, nCellsPerDim = nPointsPerDim - 1

## Bezier Curves (1D)

## B-Spline Curves (1D)
In B-Spline curves, the Bezier control points require prior or next segments, so the first and last segment are not computable.

The "thirds" points are interpolated directly from the b-spline control points.
The mid points are interpolated from the "thirds" points.
In a uniform grid (1x1 cells) the mid-points x,y are the same as the cell corner points, so we don't actually need to interpolate them.
We can't compute all points in the first edge cell, by our definition that the first point is mid-point using prior segment.
There are three bezier control points per cell (the mid from previous segment, then the two thirds points).

## Build
This uses the following other packages:
* cpp-opencv-util and cpp-base-util
* fmt
* OpenCV
* Boost (for gnuplot)

This build is not particularly portable nor self-contained yet. I am using vcpkg to manage dependencies and have no scripts yet to make that easy. So for now, you could use vcpkg or other package manager to make those available.
