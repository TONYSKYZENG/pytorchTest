cmake_minimum_required(VERSION 3.14)
project(example-app)

# Set Optimization Flags
set(CMAKE_CXX_FLAGS "-std=c++20 -Wall -Werror=return-type -Werror=unused-variable -Werror=unused-parameter")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DNO_RACE_CHECK -DIntelliStream_DEBUG_MODE=1")
set(CMAKE_CXX_FLAGS_RELEASE "-Wno-ignored-qualifiers -Wno-sign-compare -O3")
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_FLAGS "-std=c++11")
#set (CMAKE_CUDA_ARCHITECTURES 55)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_FLAGS "-std=c++11")
add_executable(torchcpp main.cpp)
message("${TORCH_LIBRARIES}")
target_link_libraries(torchcpp "${TORCH_LIBRARIES}")
set_property(TARGET torchcpp PROPERTY CXX_STANDARD 20)
#add_library(myVecAdd SHARED myVecAdd.cpp)
#target_link_libraries(myVecAdd "${TORCH_LIBRARIES}")

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/pythonTest.py
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET example-app
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:example-app>)
endif (MSVC)
