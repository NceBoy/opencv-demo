cmake_minimum_required(VERSION 3.5)
project (opencv-demo)

SET(EXECUTABLE_OUTPUT_PATH  ${PROJECT_BINARY_DIR}/out/)

if(CMAKE_BUILD_TYPE MATCHES "Debug")
set(LIB_PATH ${PROJECT_SOURCE_DIR}/lib/Debug)
FILE(GLOB_RECURSE LIBS ${LIB_PATH}/*.lib)
FILE(GLOB_RECURSE RUNTIME_LIBS ${LIB_PATH}/*.dll)
else()
set(LIB_PATH ${PROJECT_SOURCE_DIR}/lib/Release)
FILE(GLOB_RECURSE LIBS ${LIB_PATH}/*.lib)
FILE(GLOB_RECURSE RUNTIME_LIBS ${LIB_PATH}/*.dll)
endif()

include_directories(${PROJECT_SOURCE_DIR}/include)
add_executable(opencv-demo ${PROJECT_SOURCE_DIR}/opencv-demo.cpp)

add_custom_command(TARGET opencv-demo 
POST_BUILD 
COMMAND ${CMAKE_COMMAND} -E copy_if_different ${RUNTIME_LIBS} ${EXECUTABLE_OUTPUT_PATH}/${CMAKE_BUILD_TYPE}
)
target_link_libraries(opencv-demo ${LIBS})

