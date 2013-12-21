# Create a library called "Hello" which includes the source file "hello.cxx". 
# The extension is already found. Any number of sources could be listed here. 
set(PARTY3RD "${MRF_PROPOGATAION_SOURCE_DIR}/3rdParty/")
INCLUDE_DIRECTORIES(${PARTY3RD})
add_library (MRFPRop GMM.cpp  graph.cpp  imagegraph.cpp   maxflow.cpp  SLIC.cpp)

