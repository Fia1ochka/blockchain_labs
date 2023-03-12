#include <iostream>
#include <omp.h>
#include <chrono>

const int MATRIX_SIZE = 1000;

void matrixMultiplicationSequential(double** A, double** B, double** C) {
	for (int i = 0; i < MATRIX_SIZE; ++i) {
		for (int j = 0; j < MATRIX_SIZE; ++j) {
			for (int k = 0; k < MATRIX_SIZE; ++k) {
				C[i][j] += A[i][k] * B[j][k];
			}
		}
	}
}

void matrixMultiplicationParallel(double** A, double** B, double** C) {
	int i, j, k;
#pragma omp parallel for private(i, j, k) shared(A, B, C) schedule (static)
	for (i = 0; i < MATRIX_SIZE; ++i) {
		for (j = 0; j < MATRIX_SIZE; ++j) {
			for (k = 0; k < MATRIX_SIZE; ++k) {
				C[i][j] += A[i][k] * B[j][k];
			}
		}
	}
}

int main() {
	double** A = new double* [MATRIX_SIZE];
	double** B = new double* [MATRIX_SIZE];
	double** C = new double* [MATRIX_SIZE];

	for (int i = 0; i < MATRIX_SIZE; i++) {
		A[i] = new double[MATRIX_SIZE];
		B[i] = new double[MATRIX_SIZE];
		C[i] = new double[MATRIX_SIZE];
	}

	for (int i = 0; i < MATRIX_SIZE; i++) {
		for (int j = 0; j < MATRIX_SIZE; j++) {
			A[i][j] = 1;
			B[i][j] = 1;
			C[i][j] = 0;
		}
	}

	auto start = std::chrono::high_resolution_clock::now();
	matrixMultiplicationSequential(A, B, C);
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

	std::cout << "Sequential execution time: " << duration.count() << " ms" << std::endl;

	auto start_p = std::chrono::high_resolution_clock::now();
	matrixMultiplicationParallel(A, B, C);
	auto end_p = std::chrono::high_resolution_clock::now();
	auto duration_p = std::chrono::duration_cast<std::chrono::milliseconds>(end_p - start_p);

	std::cout << "Parallel execution time: " << duration_p.count() << " ms" << std::endl;

	for (int i = 0; i < MATRIX_SIZE; i++) {
		delete[] A[i];
		delete[] B[i];
		delete[] C[i];
	}
	delete[] A;
	delete[] B;
	delete[] C;

	return 0;
}
