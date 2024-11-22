#include <iostream>
#include <random>
#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  // Define matrix dimensions
  constexpr size_t N = 3;

  // Create a SYCL queue
  queue q;

  // Allocate unified shared memory (USM) for matrices
  int *matrixA = malloc_shared<int>(N * N, q);
  int *matrixB = malloc_shared<int>(N * N, q);
  int *matrixC = malloc_shared<int>(N * N, q);

  // Initialize Matrix A with random integers (1 to 100)
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(1, 100);

  for (size_t i = 0; i < N * N; ++i) {
    matrixA[i] = dis(gen);
  }

  // Initialize Matrix B with ones
  for (size_t i = 0; i < N * N; ++i) {
    matrixB[i] = 1;
  }

  // Submit a kernel to perform matrix addition: C = A + B
  q.parallel_for(range<2>(N, N), [=](id<2> idx) {
     size_t row = idx[0];
     size_t col = idx[1];
     size_t index = row * N + col;
     matrixC[index] = matrixA[index] + matrixB[index];
   }).wait();

  // Submit a kernel to perform element-wise multiplication: C = C * C
  q.parallel_for(range<2>(N, N), [=](id<2> idx) {
     size_t row = idx[0];
     size_t col = idx[1];
     size_t index = row * N + col;
     matrixC[index] *= matrixC[index];
   }).wait();

  // Print the resulting matrix
  std::cout << "Resulting Matrix C:" << std::endl;
  for (size_t row = 0; row < N; ++row) {
    for (size_t col = 0; col < N; ++col) {
      std::cout << matrixC[row * N + col] << " ";
    }
    std::cout << std::endl;
  }

  // Free the USM memory
  free(matrixA, q);
  free(matrixB, q);
  free(matrixC, q);

  return 0;
}
