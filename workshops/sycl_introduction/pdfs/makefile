# Makefile for compiling SYCL programs with icpx
# Usage:
#   make          - Compile with default SYCL backend
#   make cuda     - Compile with CUDA backend
#   make clean    - Remove compiled executable

SRC = a.cpp

CXX = icpx

CXXFLAGS = -fsycl -std=c++17

OUT = output

all: $(OUT)

$(OUT): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(OUT) $(SRC)

cuda:
	$(CXX) $(CXXFLAGS) -fsycl-targets=nvptx64-nvidia-cuda -o $(OUT) $(SRC)

clean:
	rm -f $(OUT)
