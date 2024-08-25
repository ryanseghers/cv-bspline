cmake_minimum_required (VERSION 3.20 FATAL_ERROR)
project(cv-bspline LANGUAGES CXX VERSION 0.1.0)

set (CMAKE_CXX_STANDARD 20)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

if(WIN32)
	# 7.5 is 20 series, 8.6 is 30 series, 8.9 is 40 series
	set(CMAKE_CUDA_ARCHITECTURES "75;86;89")

	enable_language(CUDA)
	find_package(CUDAToolkit)

	# Fix error: Target "CVRemap" requires the language dialect "CUDA20" (with compiler
	#  extensions).  But the current compiler "NVIDIA" does not support this, or
	#  CMake does not know the flags to enable it.
	set(CMAKE_CUDA_STANDARD 20)

	if(NOT DEFINED CMAKE_CUDA20_STANDARD_COMPILE_OPTION)
	  set(CMAKE_CUDA20_STANDARD_COMPILE_OPTION "")
	  set(CMAKE_CUDA20_EXTENSION_COMPILE_OPTION "")
	endif()

	set(CMAKE_CUDA_FLAGS "-std=c++20")

    # For VS instrumentation profiler
    # Enable these lines, then do a clean build, then run analyzer from exe.
    #add_link_options("/PROFILE")
    #set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Zi")
endif()

# other dependencies
find_package(fmt CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)

# Boost just for gnuplot wrapper
find_package(Boost REQUIRED COMPONENTS filesystem iostreams) # system utility tuple iostreams

# cvplot from my local source because have made changes
include(FetchContent)
set (CMAKE_POLICY_DEFAULT_CMP0077 NEW) # for cv-plot warnings
set (CVPLOT_WITH_TESTS OFF)
set (CVPLOT_WITH_EXAMPLES OFF)
set (CVPLOT_HEADER_ONLY OFF)
set (CVPLOT_INSTALL OFF)

include(FetchContent)

FetchContent_Declare(
	CvPlot
	GIT_REPOSITORY "https://github.com/ryanseghers/cv-plot.git"
	GIT_TAG        f18dac61  # bar plot branch
	SOURCE_SUBDIR ./CvPlot
)

FetchContent_MakeAvailable(CvPlot)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

set(SOURCE_FILES
	BezierMat.cpp
	BezierMat.h

	BezierUtil.cpp
	BezierUtil.h

	BezierVector.cpp
	BezierVector.h

	BSplineMat.cpp
	BSplineMat.h

	BSplineVector.cpp
	BSplineVector.h

	BSplineGrid.cpp
	BSplineGrid.h

	CvPlotUtil.cpp
	CvPlotUtil.h

	GnuPlotUtil.cpp
	GnuPlotUtil.h

	BSplineMiscUtil.cpp
	BSplineMiscUtil.h

	ImageTransformBSpline.h
	ImageTransformBSpline.cpp

	MatWave.cpp
	MatWave.h

	fitCurves/fitCurves.h
	fitCurves/fitCurves.cpp

	mainMiscUtil.h
	mainMiscUtil.cpp

	main.cpp
)

if(WIN32)
    list(APPEND SOURCE_FILES
	cudaUtil.h
	cudaUtil.cu

	cudaBSplineCoeffs.h
	cudaBSplineCoeffs.cu

	cudaBSplineEval.h
	cudaBSplineEval.cu

	cudaBSplineTransform.h
	cudaBSplineTransform.cu

	ShowBSplineDistortion.h
	ShowBSplineDistortion.cpp
    )
endif()

add_executable (cv-bspline-main ${SOURCE_FILES})
target_include_directories(cv-bspline-main PUBLIC ".")

target_link_libraries(cv-bspline-main PRIVATE ${OpenCV_LIBS} fmt::fmt CppOpenCVUtilLib CvPlot::CvPlot Boost::filesystem Boost::iostreams)

if(WIN32)
	set_property(TARGET cv-bspline-main PROPERTY CUDA_SEPARABLE_COMPILATION ON)
	target_link_libraries(cv-bspline-main PRIVATE CUDA::cudart_static)
endif()
