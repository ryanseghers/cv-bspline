#include <string>
#include <vector>
#include <tuple>
#include <opencv2/opencv.hpp>

#include "GnuPlotUtil.h"
#include "BSplineMiscUtil.h"
//#include "C:/Dev/gnuplot-iostream/gnuplot-iostream.h"

using namespace std;

namespace CvImageDeform
{
    void gnuPlot3dSurface(const std::string& chartTitle, const std::vector<std::vector<cv::Point3f>>& points)
    {
        // Gnuplot gp;
        // gp << "set hidden3d nooffset\n";
        // gp << "set terminal wxt size 1400,1000\n";
        // auto plots = gp.splotGroup();

        // int ydim = (int)points.size();
        // int xdim = (int)points[0].size();

        // // Hacked up using Eigen matrices because that's the example I found.
        // //Eigen::MatrixXd ptsx(ydim, xdim);
        // //Eigen::MatrixXd ptsy(ydim, xdim);
        // //Eigen::MatrixXd ptsz(ydim, xdim);

        // //for (int yi = 0; yi < ydim; yi++)
        // //{
        // //    for (int xi = 0; xi < xdim; xi++)
        // //    {
        // //        ptsx(yi, xi) = points[yi][xi].x();
        // //        ptsy(yi, xi) = points[yi][xi].y();
        // //        ptsz(yi, xi) = points[yi][xi].z();
        // //    }
        // //}

        // //plots.add_plot2d(tuple{ptsx, ptsy, ptsz}, "with lines title '" + chartTitle + "'");

        // //for (int yi = 0; yi < ydim; yi++)
        // //{
        // //    for (int xi = 0; xi < xdim; xi++)
        // //    {
        // //        fmt::print("({0:.1f}, {1:.1f}, {2:.1f}), ", ptsx(yi, xi), ptsy(yi, xi), ptsz(yi, xi));
        // //    }
        // //    fmt::print("\n");
        // //}

        // std::vector<std::vector<std::tuple<float,float,float>>> pts(ydim);

        // for(int yi=0; yi<ydim; yi++)
        // {
        //     pts[yi].resize(xdim);

        //     for(int xi=0; xi<xdim; xi++)
        //     {
        //         pts[yi][xi] = std::make_tuple(
        //             points[yi][xi].x,
        //             points[yi][xi].y,
        //             points[yi][xi].z
        //         );
        //     }
        // }

        // plots.add_plot2d(pts, "with lines title '" + chartTitle + "'");

        // //for (int yi = 0; yi < ydim; yi++)
        // //{
        // //    for (int xi = 0; xi < xdim; xi++)
        // //    {
        // //        fmt::print("({0:.1f}, {1:.1f}, {2:.1f}), ", std::get<0>(pts[yi][xi]), std::get<1>(pts[yi][xi]), std::get<2>(pts[yi][xi]));
        // //    }
        // //    fmt::print("\n");
        // //}

        // gp << plots;
        // std::cout << "Press enter to exit." << std::endl;
        // std::cin.get();
    }

    // This is one of their examples.
    void tryGnuPlot()
    {
        // Gnuplot gp;
        // gp.width(1400);

        // std::vector<std::pair<double, double> > xy_pts_A;
        // for(double x=-2; x<2; x+=0.01) {
        //     double y = x*x*x;
        //     xy_pts_A.push_back(std::make_pair(x, y));
        // }

        // std::vector<std::pair<double, double> > xy_pts_B;
        // for(double alpha=0; alpha<1; alpha+=1.0/24.0) {
        //     double theta = alpha*2.0*3.14159;
        //     xy_pts_B.push_back(std::make_pair(cos(theta), sin(theta)));
        // }

        // gp << string("set xrange [-2:2]\nset yrange [-2:2]\n");

        // // Data will be sent via a temporary file.  These are erased when you call
        // // gp.clearTmpfiles() or when gp goes out of scope.  If you pass a filename
        // // (e.g. "gp.file1d(pts, 'mydata.dat')"), then the named file will be created
        // // and won't be deleted (this is useful when creating a script).
        // gp << "plot" << gp.file1d(xy_pts_A) << "with lines title 'cubic',"
        //     << gp.file1d(xy_pts_B) << "with points title 'circle'" << std::endl;

        // // For Windows, prompt for a keystroke before the Gnuplot object goes out of scope so that
        // // the gnuplot window doesn't get closed.
        // std::cout << "Press enter to exit." << std::endl;
        // std::cin.get();
    }

    void gnuPlot3dSurfaceZs(const std::string& chartTitle, const cv::Mat& zs)
    {
        gnuPlot3dSurface(chartTitle, matToPoints(zs));
    }

    void gnuPlot3dSurfaceMat(const std::string& chartTitle, const cv::Mat& m)
    {
        gnuPlot3dSurface(chartTitle, matToPoints(m));
    }
}
