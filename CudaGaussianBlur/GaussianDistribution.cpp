#include "GaussianDistribution.h"

#define _USE_MATH_DEFINES

#include<cmath>
#include<iomanip>
#include<iostream>


namespace Distribution
{
	void Generate(const int size, double** kernel, double stddev)
	{
		// Check Variables ==============================================================

		if (size <= 0 || stddev <= 0) {
			std::cerr << "Error: size and stddev must be positive.\n";
			return;
		}


		// Memory Allocation ============================================================

		*kernel = (double*)malloc(sizeof(double) * size);
		if (!*kernel)
		{
			std::cerr << "Error : kernel memory allocation failure\n";
			return;
		}


		// Generate Nums ================================================================

		double sum = 0.0;
		double variance = stddev * stddev;
		int center = size / 2;

		for (int i = 0; i < size; ++i)
		{
			double diff = static_cast<double>(i - center);
			double exponent = -(diff * diff) / (2.0 * variance);
			double coefficient = 1.0 / (std::sqrt(1.0 * M_PI * stddev));
			double value = coefficient * std::exp(exponent);

			(*kernel)[i] = value;
			sum += value;
		}


		// Check sum ====================================================================

		if (sum == 0.0) {
			std::cerr << "Error: Sum of kernel values is zero. Check stddev and size.\n";
			free(*kernel);
			return;
		}


		// Normalization ===============================================================

		for (int i = 0; i < size; ++i)
		{
			(*kernel)[i] /= sum;
		}
	}
}