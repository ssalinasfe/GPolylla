cmake_minimum_required(VERSION 3.0.0)
project(PolyllaCUDA VERSION 0.1.0)
project(cmake_cuda LANGUAGES CUDA)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch")

add_subdirectory(src)
include_directories(src)

add_executable(Polylla main.cu)

target_link_libraries(Polylla PUBLIC meshfiles)
set_target_properties(meshfiles PROPERTIES LINKER_LANGUAGE CXX)
