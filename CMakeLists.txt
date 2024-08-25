cmake_minimum_required (VERSION 3.20 FATAL_ERROR)
project(cv-bspline LANGUAGES CXX VERSION 0.1.0)

# cpp-opencv-util adds cpp-base-util
add_subdirectory(cpp-opencv-util)
add_subdirectory(src)
