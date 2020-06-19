cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # Produce compile_commands.json
project(Faces)
set(LIBRARY_TARGET_NAME ${PROJECT_NAME})
find_package(Torch REQUIRED)
find_package(Python REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(TorchVision REQUIRED)
find_package( OpenCV REQUIRED )

include_directories(${PYTHON_INCLUDE_DIRS})

set(${LIBRARY_TARGET_NAME}_SRC
    PostProcessing.cpp
    PreProcessing.cpp
)

set(${LIBRARY_TARGET_NAME}_HDR
    DataProcessing.hpp
)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3 -O0")
if (UNIX)
  add_library(${LIBRARY_TARGET_NAME} SHARED ${${LIBRARY_TARGET_NAME}_SRC})
endif(UNIX)
target_link_libraries(${LIBRARY_TARGET_NAME} PUBLIC "${TORCH_LIBRARIES}" TorchVision::TorchVision ${PYTHON_LIBRARIES} ${OpenCV_LIBS})
target_compile_options(${LIBRARY_TARGET_NAME} PUBLIC)

add_executable(example-app main.cpp)
target_link_libraries(example-app  PUBLIC Faces "${TORCH_LIBRARIES}" TorchVision::TorchVision ${OpenCV_LIBS})
set_property(TARGET example-app PROPERTY CXX_STANDARD 17)
