cmake_minimum_required(VERSION 3.1)
project(Inpaint)

include_directories(libigl/ libigl/eigen/)
add_executable(Inpaint main.cpp)

