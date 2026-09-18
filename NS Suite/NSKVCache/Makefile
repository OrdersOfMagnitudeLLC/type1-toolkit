CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall

all: kv_box kv_alloc

kv_box: kv_box.cpp
	$(CXX) $(CXXFLAGS) -o kv_box kv_box.cpp

kv_alloc: kv_alloc.cpp
	$(CXX) $(CXXFLAGS) -o kv_alloc kv_alloc.cpp

clean:
	rm -f kv_box kv_alloc

.PHONY: clean
