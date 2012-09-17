CC=g++
OPENCVLIBS=$(shell pkg-config opencv --cflags --libs)$
CFLAGS=-c -Wno-cpp
LDFLAGS=  -Wno-cpp
LIBS= -lboost_regex -lboost_filesystem -lboost_system  $(OPENCVLIBS)
SOURCES=GMM.cpp  graph.cpp  imagegraph.cpp  main.cpp  maxflow.cpp  SLIC.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=segmenter

all: $(SOURCES) $(EXECUTABLE)
	
$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(LDFLAGS) $(OBJECTS) -o $(EXECUTABLE) $(LIBS)

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -rf *.o *.png *.txt segmenter
