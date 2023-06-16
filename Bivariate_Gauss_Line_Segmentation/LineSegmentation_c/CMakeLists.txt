cmake_minimum_required(VERSION 3.5.0)
project(LineSegmentation_c)
set(OpenCV_DIR "C:/OpenCV/opencv/build")
set(OpenCV_INCLUDE_DIRS "C:/OpenCV/opencv/build/include")

find_package(OpenCV REQUIRED)

file(GLOB MY_SOURCES *.hpp *.cpp)
file(GLOB MY_HEADER *.hpp)

message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( ${PROJECT_NAME} ${MY_SOURCES} ${MY_HEADER})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})