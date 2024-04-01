# cv-bspline
C++ OpenCV Bezier and B-Spline Library

## Introduction
This has Bezier and B-spline curve (1-D) and surface (2-D) implementations in C++ using OpenCV, and in CUDA.

## Background and Definitions
In this implementation, in the 1-D case, we picture the B-spline control points as defining a sequence of segments. Note that the last point in the sequence only ends a segment, it doesn't define another segment.

And in the 2-D case, we picture the B-spline control points as defining a grid of cells, where again the last row and column of points only end cells, they don't define more cells.

So far this always uses a uniform control point grid, meaning the grid composed by the control points has cells that are all 1x1. This simplifies the implementation and increases the performance.
If the B-spline control points are on a uniform grid, then the Bezier points also end up on a uniform grid.

The final implementations use cv::Mat. I started with std::vector and Eigen3 implementations (the *Vector.h/cpp files) then switched to cv::Mat.

## Bezier Curves (1D)
A Bezier curve is defined by 4 control points, and thus a surface is defined by 4x4 control points.

## B-Spline Curves (1D)
B-spline control points provide a way to join multiple Bezier segments/surface patches by computing Bezier control points such that adjacent Bezier segments/cells are continuous.

Each Bezier segment can be viewed as having 3 control points and getting its 4th control point from the next segment.
Thus each Bezier cell can be viewed as having 3x3 control points and getting its 4th row and column of control points from the next cells.

In B-Spline curves, computing the Bezier control points requires prior or next segments/cells, so the first and last segments/cells are not computable.
Thus for surfaces, the edge cells are not computable.

(See jpeg whiteboard image for context.)
The "thirds" points are interpolated directly from the b-spline control points.
The mid points are interpolated from the "thirds" points.
In a uniform grid (1x1 cells) the Bezier control point x,y values are uniform and trivial, so we don't actually need to interpolate them.
There are three bezier control points per segment/cell (the mid from previous segment/cell, then the two thirds points) and its fourth is shared with the next segment/cell.

## Build
This uses the following other packages:
* cpp-opencv-util and cpp-base-util
* fmt
* OpenCV
* cv-plot (for debug curve plots)
* gnuplot-iostream (for debug surface plots) 
* Boost (for gnuplot-iostream)

This build is not particularly portable nor self-contained yet. I am using vcpkg to manage dependencies and have no scripts yet to make that easy. So for now, you could use vcpkg or other package manager to make those available.

## Code Design Notes
There are some intersecting facts about array/matrix indexing that can make things surprisingly tricky.

In this section we'll talk about the 1-D case, but the issues trivially extend to the 2-D case.

As an example, let's take 5 b-spline control points:
b-spline control points     0     1     2     3     4
Bezier control points       a b c d e f g h i j k l m

Those 5 b-spline points define 4 segments. 

Since the canonical algorithm requires a prior segment to compute the first Bezier control point, the first Bezier control point (a) is not computable and thus the first segment is not computable. Similarly for the last segment, the last Bezier control point (m) is not computable. So those 5 b-spline control points define 4 segments, 2 of which are strictly computable.

Each segment requires 4 Bezier control points to compute, but a point is shared between segments, so we have 3 Bezier control points per segment. To compute the last computable segment, we need the next segment's first Bezier control point. Thus we actually need 3 points per segment plus one.

To restate:
In this example our 5 b-spline control points define 4 segments, so we might represent this as 4 segments in code.
But 4 segments only yield 2 computable (non-edge) segments, so we might represent this as 2 segments in code.
And each segment requires 3 Bezier control points, so we might represent this as two sets of 3 points in code
But a point is shared between segments, so the last segment requires one more point, so we need 3 sets of 3 points in code (the last being partial) to represent the 2 computable segments.

In the code, if we don't represent the first segment in arrays/data structures because it isn't computable, then we have to keep adjusting for that, so I chose to represent the first segment even if we consider it not strictly computable. And then later I decided to treat it as computable anyway, see below.

Non-Strictly Computable Segments
The first point (a) is not strictly computable, but it seems a reasonable approximation to set it to the first b-spline control point's value, and so I have chosen to do that. Thus the first segment is also computable, if not strictly. Note that the caller can simply choose to not use the first segment if they don't want to.

Similarly for the last segment, the last point (m) is not strictly computable, but we can trivially set it to the last b-spline control point, and so I have chosen to do that. 

This affects data structures and indexing because if we store 3 points per segment, then point m is in a final, 5th, segment.
And thus for 5 b-spline control points we have 4 segments, all of which are (non-strictly) computable, requiring storing 5 segments worth of points (because the code generally works in units of segments). 
