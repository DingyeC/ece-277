set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )

add_subdirectory(pybind11_cuda_madd)
add_subdirectory(pybind11_cuda_mmul)
add_subdirectory(pybind11_cuda_mmac)
