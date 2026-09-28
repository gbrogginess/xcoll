# Build of the thin-gas-target benchmark against Geant4 (FTFP_BERT).
#   mkdir build && cd build && cmake .. && make
cmake_minimum_required(VERSION 3.16)
project(beamgas_benchmark CXX)
find_package(Geant4 REQUIRED)
include(${Geant4_USE_FILE})
add_executable(beamgas_benchmark beamgas_benchmark.cc)
target_link_libraries(beamgas_benchmark ${Geant4_LIBRARIES})
