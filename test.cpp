/* Test program for cubature.
 *
 * Copyright (c) 2005-2013 Steven G. Johnson
 *
 * Portions (see comments) based on HIntLib (also distributed under
 * the GNU GPL, v2 or later), copyright (c) 2002-2005 Rudolf Schuerer.
 *     (http://www.cosy.sbg.ac.at/~rschuer/hintlib/)
 *
 * Portions (see comments) based on GNU GSL (also distributed under
 * the GNU GPL, v2 or later), copyright (c) 1996-2000 Brian Gough.
 *     (http://www.gnu.org/software/gsl/)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 */

// Usage: ./test <dim> <tol> <integrand> <maxeval>
//
// where <dim> = # dimensions, <tol> = relative tolerance, <integrand> is either
// 0/1/2 for the three test integrands (see below), and <maxeval> is the maximum
// # function evaluations (0 for none).

#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

#include <cubature.hpp>

int count = 0;
int which = 0;
double const RADIUS = 0.50124145262344534123412;
double const K_2_SQRTPI = 1.12837916709551257390;
double const K_PI = 3.14159265358979323846;

// Simple constant function.
template<std::size_t D>
double fconst(cubature::Point<D, double> x) {
	return 1;
}

// f0, f1, f2, and f3 are test functions from the Monte-Carlo integration
// routines in GSL 1.6 (monte/test.c).  Copyright (c) 1996-2000 Michael Booth,
// GNU GPL.

// Simple product function.
template<std::size_t D>
double f0(cubature::Point<D, double> x) {
	double prod = 1;
	for (std::size_t d = 0; d < D; ++d) {
		prod *= 2 * x[d];
	}
	return prod;
}

// Gaussian centered at 1/2.
template<std::size_t D>
double f1(cubature::Point<D, double> x) {
	double a = 0.1;
	double sum = 0;
	for (std::size_t d = 0; d < D; ++d) {
		double dx = x[d] - 0.5;
		sum += dx * dx;
	}
	return std::pow(K_2_SQRTPI / (2. * a), D) * std::exp(-sum / (a * a));
}

// Double Gaussian.
template<std::size_t D>
double f2(cubature::Point<D, double> x) {
	double a = 0.1;
	double sum1 = 0;
	double sum2 = 0;
	for (std::size_t d = 0; d < D; ++d) {
		double dx1 = x[d] - 1. / 3.;
		double dx2 = x[d] - 2. / 3.;
		sum1 += dx1 * dx1;
		sum2 += dx2 * dx2;
	}
	return 0.5 * std::pow(K_2_SQRTPI / (2. * a), D) * (std::exp(-sum1 / (a * a)) + std::exp(-sum2 / (a * a)));
}

// Tsuda's example.
template<std::size_t D>
double f3(cubature::Point<D, double> x) {
	double c = (1.0 + std::sqrt(10.0)) / 9.0;
	double prod = 1;
	for (std::size_t d = 0; d < D; ++d) {
		prod *= c / (c + 1) * std::pow((c + 1) / (c + x[d]), 2);
	}
	return prod;
}

// Test integrand from W. J. Morokoff and R. E. Caflisch, "Quasi-Monte Carlo
// integration," J. Comput. Phys 122, 218-230 (1995). Designed for integration
// on [0,1]^dim, integral = 1.
template<std::size_t D>
double morokoff(cubature::Point<D, double> x) {
	double p = 1. / D;
	double prod = std::pow(1 + p, D);
	for (std::size_t d = 0; d < D; ++d) {
		prod *= pow(x[d], p);
	}
	return prod;
}

template<std::size_t D>
double f_test(cubature::Point<D, double> x) {
	++count;
	double val;
	switch (which) {
	case 0:
		// Simple smooth (separable) objective: prod. cos(x[d]).
		val = 1;
		for (std::size_t d = 0; d < D; ++d) {
			val *= std::cos(x[d]);
		}
		break;
	case 1:
		{
			// Integral of exp(-x^2), rescaled to (0,infinity) limits.
			double scale = 1;
			val = 0;
			for (std::size_t d = 0; d < D; ++d) {
				if (x[d] > 0) {
					double z = (1 - x[d]) / x[d];
					val += z * z;
					scale *= K_2_SQRTPI / (x[d] * x[d]);
				}
				else {
					scale = 0;
					break;
				}
			}
			val = exp(-val) * scale;
			break;
		}
	case 2:
		// Discontinuous objective: volume of hypersphere.
		val = 0;
		for (std::size_t d = 0; d < D; ++d) {
			val += x[d] * x[d];
		}
		val = (val < RADIUS * RADIUS) ? 1 : 0;
		break;
	case 3:
		val = f0(x);
		break;
	case 4:
		val = f1(x);
		break;
	case 5:
		val = f2(x);
		break;
	case 6:
		val = f3(x);
		break;
	case 7:
		val = morokoff(x);
		break;
	default:
		std::cout << "Unknown integrand " << which << '\n';
		exit(EXIT_FAILURE);
	}
	return val;
}

// Surface area of n-dimensional unit hypersphere.
template<std::size_t D>
double S() {
	double val;
	int fact = 1;
	std::size_t n = D;
	if (n % 2 == 0) {
		val = 2 * std::pow(K_PI, n * 0.5);
		n = n / 2;
		while (n > 1) {
			fact *= (n -= 1);
		}
		val /= fact;
	}
	else {
		val = (1 << (n / 2 + 1)) * std::pow(K_PI, n / 2);
		while (n > 2) {
			fact *= (n -= 2);
		}
		val /= fact;
	}
	return val;
}

template<std::size_t D>
double exact_integral(cubature::Point<D, double> xmax) {
	double val;
	switch(which) {
	case 0:
		val = 1;
		for (std::size_t d = 0; d < D; ++d) {
			val *= std::sin(xmax[d]);
		}
		break;
	case 2:
		val = (D == 0) ? 1 : S<D>() * std::pow(RADIUS * 0.5, D) / D;
		break;
	default:
		val = 1;
	}
	return val;
}

int main(int argc, char **argv) {
	if (argc <= 1) {
		std::cout << "Usage: " << argv[0] << " [dim] [reltol] [maxeval] [integrand]\n";
		return EXIT_FAILURE;
	}

	std::size_t dim = (argc > 1) ? std::stoi(argv[1]) : 2;
	double tol = (argc > 2) ? std::stof(argv[2]) : 1e-2;
	std::size_t max_eval = (argc > 3) ? std::stoi(argv[3]) : 0;
	which = (argc > 4) ? std::stoi(argv[4]) : 0;

	cubature::EstErr<double> est_err;
	double exact;
	switch (dim) {
	case 0:
		{
			auto xmin = cubature::Point<0, double>{};
			auto xmax = cubature::Point<0, double>{};
			est_err = cubature::cubature(&f_test<0>, xmin, xmax, max_eval, 0., tol);
			exact = exact_integral(xmax);
			break;
		}
	case 1:
		{
			auto xmin = cubature::Point<1, double>{ 0 };
			auto xmax = cubature::Point<1, double>{ 1 };
			est_err = cubature::cubature(&f_test<1>, xmin, xmax, max_eval, 0., tol);
			exact = exact_integral(xmax);
			break;
		}
	case 2:
		{
			auto xmin = cubature::Point<2, double>{ 0, 0 };
			auto xmax = cubature::Point<2, double>{ 1, 1 };
			est_err = cubature::cubature(&f_test<2>, xmin, xmax, max_eval, 0., tol);
			exact = exact_integral(xmax);
			break;
		}
	case 3:
		{
			auto xmin = cubature::Point<3, double>{ 0, 0, 0 };
			auto xmax = cubature::Point<3, double>{ 1, 1, 1 };
			est_err = cubature::cubature(&f_test<3>, xmin, xmax, max_eval, 0., tol);
			exact = exact_integral(xmax);
			break;
		}
	case 4:
		{
			auto xmin = cubature::Point<4, double>{ 0, 0, 0, 0 };
			auto xmax = cubature::Point<4, double>{ 1, 1, 1, 1 };
			est_err = cubature::cubature(&f_test<4>, xmin, xmax, max_eval, 0., tol);
			exact = exact_integral(xmax);
			break;
		}
	case 5:
		{
			auto xmin = cubature::Point<5, double>{ 0, 0, 0, 0, 0 };
			auto xmax = cubature::Point<5, double>{ 1, 1, 1, 1, 1 };
			est_err = cubature::cubature(&f_test<5>, xmin, xmax, max_eval, 0., tol);
			exact = exact_integral(xmax);
			break;
		}
	case 6:
		{
			auto xmin = cubature::Point<6, double>{ 0, 0, 0, 0, 0, 0 };
			auto xmax = cubature::Point<6, double>{ 1, 1, 1, 1, 1, 1 };
			est_err = cubature::cubature(&f_test<6>, xmin, xmax, max_eval, 0., tol);
			exact = exact_integral(xmax);
			break;
		}
	case 7:
		{
			auto xmin = cubature::Point<7, double>{ 0, 0, 0, 0, 0, 0, 0 };
			auto xmax = cubature::Point<7, double>{ 1, 1, 1, 1, 1, 1, 1 };
			est_err = cubature::cubature(&f_test<7>, xmin, xmax, max_eval, 0., tol);
			exact = exact_integral(xmax);
			break;
		}
	case 8:
		{
			auto xmin = cubature::Point<8, double>{ 0, 0, 0, 0, 0, 0, 0, 0 };
			auto xmax = cubature::Point<8, double>{ 1, 1, 1, 1, 1, 1, 1, 1 };
			est_err = cubature::cubature(&f_test<8>, xmin, xmax, max_eval, 0., tol);
			exact = exact_integral(xmax);
			break;
		}
	case 9:
		{
			auto xmin = cubature::Point<9, double>{ 0, 0, 0, 0, 0, 0, 0, 0, 0 };
			auto xmax = cubature::Point<9, double>{ 1, 1, 1, 1, 1, 1, 1, 1, 1 };
			est_err = cubature::cubature(&f_test<9>, xmin, xmax, max_eval, 0., tol);
			exact = exact_integral(xmax);
			break;
		}
	case 10:
		{
			auto xmin = cubature::Point<10, double>{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
			auto xmax = cubature::Point<10, double>{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
			est_err = cubature::cubature(&f_test<10>, xmin, xmax, max_eval, 0., tol);
			exact = exact_integral(xmax);
			break;
		}
	default:
		std::cout << "Only up to dimension 10 is supported.\n";
		return EXIT_FAILURE;
	}

	std::cout << dim << "-dim integral, tolerance = " << tol << '\n';
	double true_err = std::fabs(est_err.val - exact);
	std::cout << "Results: "
		<< "integral = " << std::scientific << est_err.val << ", "
		<< "est err = " << std::scientific << est_err.err << ", "
		<< "true err = " << std::scientific << true_err << '\n';
	std::cout << "# of evals = " << count << '\n';

	return EXIT_SUCCESS;
}

