#include <iostream>
#include <chrono>
#include <vector>

bool checkSolution(std::vector<double> x, std::vector<double> x_generated) {
	for (int i = 0; i < x.size(); i++) {
		if (abs(x_generated[i] - x[i]) > 0.01)
			return false;
	}

	return true;
}

int main() {
	int n = 1000;

	std::vector<std::vector<double>> a(n);
	std::vector<double> b(n);
	std::vector<double> x(n);
	std::vector<double> x_generated(n);

	for (int i = 0; i < n; i++)
	{
		a[i] = std::vector<double>(n);
		for (int j = 0; j < n; j++)
		{
			a[i][j] = rand() % 10 + 1;
		}
		x_generated[i] = rand() % 10 + 1;
	}

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			b[i] += a[i][j] * x_generated[j];
		}
	}

	auto start = std::chrono::high_resolution_clock::now();

	for (int k = 0; k < n; k++)
	{
#pragma omp parallel for
		for (int i = k + 1; i < n; i++) {
			double d = a[i][k] / a[k][k];
			for (int j = k; j < n; j++) {
				a[i][j] -= d * a[k][j];
			}
			b[i] -= d * b[k];
		}
	}

	for (int k = n - 1; k >= 0; k--)
	{
		double f = 0;

#pragma omp parallel for reduction(+:f)
		for (int j = k + 1; j < n; j++)
		{
			f += a[k][j] * x[j];
		}

		x[k] = (b[k] - f) / a[k][k];
	}


	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

	std::cout << "Execution time: " << duration.count() << " ms" << std::endl;
	std::cout << "Answer correct: " << checkSolution(x, x_generated) << std::endl;

	return 0;
}