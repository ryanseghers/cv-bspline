# cv-bspline
C++ OpenCV Bezier and B-Spline Library

## Introduction
This has Bezier and B-spline curve (1-D) and surface (2-D) implementations in C++ using OpenCV, and also CUDA implementations.

## Basics
The Bezier curve function computes points (x,y) as a parametric function (t from 0.0 to 1.0) of four control points. At t=0 the output point is the first control point, and at t=1 the output point is the final control point. The curve doesn't intersect the two middle control points.

It is important to note that the Bezier curve doesn't compute y as a function of x, it computes both as a function of t.

A B-spline is a way to compute a list of Bezier control points such that each segment of Bezier curve joins to the next, creating a single smooth curve. If each pair of B-spline control points defines a segment, then the B-spline algorithm provides a way to compute the four Bezier curve control points for that segment. Note that both the B-spline and the Bezier algorithms use points outside of the single segment in question.

The B-spline algorithm to compute Bezier control points for a segment is as follows, where the four B-spline controls points are b[0], b[1], b[2], and b[3]:
1) The first Bezier control point is linearly interpolated 2/3 of the way between b[0] and b[1]
2) The second Bezier control point is linearly interpolated 1/3 of the way between b[1] and b[2]
3) The third Bezier control point is linearly interpolated 2/3 of the way between b[1] and b[2]
4) The fourth Bezier control point is linearly interpolated 1/3 of the way between b[2] and b[3]

## Distortion Modeling
The 2-d (surface) implementation can be used as a way to represent a field over an x,y surface. For example it can be used to represent one dimension of an offset vector that represents the distortion of an image. In that case usually the b-spline grid would be lower resolution than the image, but provides a way to compute a value at arbitrary higher resolution locations in the image. That value can be used as one dimension of an offset vector that describes how that pixel is "moved" by the distortion. Two such b-spline grids (one for x and one for y) can be used to represent a 2-d smooth distortion of an image by dx and dy fields.

In the 1-d scenario, if you want to have a way to compute a y for each x, since the underlying function computes (x,y) over t, you have to somehow map the input x to a t value. Those values don't directly map to fractions between b-spline points because they actually map to bezier control points and the bezier control points are thirds points into prior and next segments. TODO

When using a 2-d grid to represent a field, note that the b-spline computation both takes and yields 3-d points.

In a volume, to represent 3-d distortion, one needs three b-spline volumes, one each for dx, dy, and dz. Thus at any arbitrary higher resolution point in the volume one can compute a vector offset (dx, dy, dz) representing the distortion of that voxel.

## Background and Definitions
In this implementation, in the 1-D case, we picture the B-spline control points as defining a sequence of segments. Note that the last point in the sequence only ends a segment, it doesn't define another segment.

And in the 2-D case, we picture the B-spline control points as defining a grid of cells, where again the last row and column of points only end cells, they don't define more cells.

So far this always uses a uniform control point grid, meaning the grid composed by the control points has cells that are all 1x1. This simplifies the implementation and increases the performance. If the B-spline control points are on a uniform grid, then the Bezier points also end up on a uniform grid.

Note that if the control points are on a uniform grid then one doesn't need the x,y values for each point as those can be trivially computed. Thus, if the grid is separately defined by origin and resolution/dims, then regardless of 1-d, 2-d, or 3-d case we only need a single scalar value per control point.

The final implementations use cv::Mat. I started with std::vector and Eigen3 implementations (the *Vector.h/cpp files) then switched to cv::Mat.

## Bezier Curves and Surfaces
A Bezier curve is defined by 4 control points, and thus a surface is defined by 4x4 control points.

## B-Spline Curves (1D) Implementation
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

## Building
This uses vcpkg to manage dependencies. You can install vcpkg and then run `vcpkg install` in the top directory of this project (the one containing this readme) to build dependencies. Then, there are different ways to tell cmake how to find the dependencies, and this currently provides a CMakePresets.json file that specifies the path to the vcpkg.cmake file using an environment variable VCPKG_ROOT. You can set this environment variable to the path to your vcpkg installation, or you can edit the CMakePresets.json file to specify the path to your vcpkg installation.

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

## Non-Strictly Computable Segments
The first point (a) is not strictly computable, but it seems a reasonable approximation to set it to the first b-spline control point's value, and so I have chosen to do that. Thus the first segment is also computable, if not strictly. Note that the caller can simply choose to not use the first segment if they don't want to.

Similarly for the last segment, the last point (m) is not strictly computable, but we can trivially set it to the last b-spline control point, and so I have chosen to do that.

This affects data structures and indexing because if we store 3 points per segment, then point m is in a final, 5th, segment.
And thus for 5 b-spline control points we have 4 segments, all of which are (non-strictly) computable, requiring storing 5 segments worth of points (because the code generally works in units of segments).

## Benchmarks

# CPU Throughput in Transformations per Second

Bicubic sampling.
On a 13900K (underclocked due to stability issues):

nThreads, transformsPerSecond
1, 26.2
2, 51.7
3, 72.4
4, 89.5
5, 102.1
6, 112.6
7, 122.2
8, 132.5
9, 135.3
10, 138.0
11, 139.9
12, 141.9
13, 144.2
14, 144.4
15, 145.8
16, 145.6
17, 146.9
18, 147.2
19, 146.8
20, 147.6
21, 147.8
22, 147.8
23, 148.9
24, 147.5

The equivalent op on my RTX 4090 is 640 transforms per second, and multiple threads do not improve throughput.
The throughput is not different between nearest-neighbor sampling and bicubic sampling.
