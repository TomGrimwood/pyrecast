# pyrecast: Recast (recastnavigation, zlib), fetched at a pinned release and built with a small
# nanobind module, `pyrecast._core`.
cmake_minimum_required(VERSION 3.18)
project(pyrecast LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

include(FetchContent)
FetchContent_Declare(
  recastnavigation
  URL https://github.com/recastnavigation/recastnavigation/archive/refs/tags/v1.6.0.tar.gz
  URL_HASH SHA256=d48ca0121962fa0639502c0f56c4e3ae72f98e55d88727225444f500775c0074
)
set(RECASTNAVIGATION_DEMO OFF CACHE BOOL "" FORCE)
set(RECASTNAVIGATION_TESTS OFF CACHE BOOL "" FORCE)
set(RECASTNAVIGATION_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(recastnavigation)

find_package(Python 3.12 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)
nanobind_add_module(_core NB_STATIC cpp/_core.cpp)
target_link_libraries(_core PRIVATE Recast)
# Only this is installed into the wheel (pyproject.toml's install.components): Recast's own
# install rules (its headers, static libraries, CMake and pkg-config files) are left out.
install(TARGETS _core LIBRARY DESTINATION pyrecast COMPONENT python)
