cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(MRF_PROPOGATAION)

FIND_PACKAGE( OpenCV REQUIRED )
add_executable (segmenter src/main.cpp)

add_subdirectory (src) 
include_directories (${MRF_PROPOGATAION_SOURCE_DIR}/src) 
include_directories (${MRF_PROPOGATAION_BINARY_DIR}/src) 

set(PARTY3RD "${MRF_PROPOGATAION_SOURCE_DIR}/3rdParty/")
INCLUDE_DIRECTORIES(${PARTY3RD})

TARGET_LINK_LIBRARIES(segmenter MRFPRop boost_regex boost_filesystem boost_system ${OpenCV_LIBS})

