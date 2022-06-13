file(GLOB HEADER_FILES *.cu)

add_library(meshfiles STATIC ${HEADER_FILES})
