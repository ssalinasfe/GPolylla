cmake_minimum_required(VERSION 3.0.0)
project(Polylla VERSION 0.1.0)
project(cmake_cuda LANGUAGES CUDA)

add_subdirectory(src)
include_directories(src)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -lnvidia-ml -lpthread -lcudart")

add_executable(Polylla main.cu)

target_link_libraries(Polylla PUBLIC meshfiles)
set_target_properties(meshfiles PROPERTIES LINKER_LANGUAGE CXX)
