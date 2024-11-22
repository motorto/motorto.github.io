#include <iostream>
#include <random>
#include <sycl/sycl.hpp>

int main() {
  constexpr size_t N = 256; // Matrix dimensions

  queue q;

  // TODO: Buffers for Matrix A, B, and C using range 2
  buffer<int, 2> matA(range<2>(N, N));

  // Initialize Matrix A with random integers and Matrix B with ones
  {
    host_accessor a_acc(matA, write_only);
    host_accessor b_acc(matB, write_only);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 100);

    for (size_t i = 0; i < N; i++) {
      for (size_t j = 0; j < N; j++) {
        a_acc[i][j] = dis(gen);
        b_acc[i][j] = 1;
      }
    }
  }

  // Kernel to perform matrix addition (C = A + B)
  q.submit([&](handler &cgh) {
     // TODO: accessors
     cgh.parallel_for(range<2>(N, N), [=](id<2> idx) {
       c_acc[idx] = // TODO:
     });
   }).wait();

  // Allocate USM memory for Matrix C
  // int *usm_matC = ;

  // Copy Matrix C buffer data to USM memory
  q.submit([&](handler &cgh) {
     // TODO:
   }).wait();

  // TODO: Kernel to perform element-wise multiplication on Matrix C (C = C * 2)
  q.submit([&](handler &cgh) {}).wait();

  // Print the resulting Matrix C
  std::cout << "Resulting Matrix C (after addition and element-wise "
               "multiplication):"
            << std::endl;
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j++) {
      std::cout << usm_matC[i * N + j] << " ";
    }
    std::cout << std::endl;
  }

  free(usm_matC, q);

  return 0;
}
