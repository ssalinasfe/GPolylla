cmake_minimum_required(VERSION 3.10)
project(cmake_cuda LANGUAGES CUDA)

project(GPolylla)

add_subdirectory(external src)
include_directories(external src)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -lnvidia-ml -lpthread -lcudart")

add_executable(GPolylla main.cu)

target_link_libraries(GPolylla PUBLIC malloccountfiles )
set_target_properties(GPolylla PROPERTIES LINKER_LANGUAGE CXX)