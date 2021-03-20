/* Adaptive multidimensional integration of a vector of integrands.
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

#ifndef CUBATURE_IPP
#define CUBATURE_IPP

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

namespace cubature {
namespace internal {

/* Adaptive multidimensional integration on hypercubes (or, really,
   hyper-rectangles) using cubature rules.

   A cubature rule takes a function and a hypercube and evaluates
   the function at a small number of points, returning an estimate
   of the integral as well as an estimate of the error, and also
   a suggested dimension of the hypercube to subdivide.

   Given such a rule, the adaptive integration is simple:

   1) Evaluate the cubature rule on the hypercube(s).
   Stop if converged.

   2) Pick the hypercube with the largest estimated error,
   and divide it in two along the suggested dimension.

   3) Goto (1).

   The basic algorithm is based on the adaptive cubature described in

   A. C. Genz and A. A. Malik, "An adaptive algorithm for numeric
   integration over an N-dimensional rectangular region,"
   J. Comput. Appl. Math. 6 (4), 295-302 (1980).

   and subsequently extended to integrating a vector of integrands in

   J. Berntsen, T. O. Espelid, and A. Genz, "An adaptive algorithm
   for the approximate calculation of multiple integrals,"
   ACM Trans. Math. Soft. 17 (4), 437-451 (1991).

   Note, however, that we do not use any of code from the above authors
   (in part because their code is Fortran 77, but mostly because it is
   under the restrictive ACM copyright license).  I did make use of some
   GPL code from Rudolf Schuerer's HIntLib and from the GNU Scientific
   Library as listed in the copyright notice above, on the other hand.

   I am also grateful to Dmitry Turbiner <dturbiner@alum.mit.edu>, who
   implemented an initial prototype of the "vectorized" functionality
   for evaluating multiple points in a single call (as opposed to
   multiple functions in a single call).  (Although Dmitry implemented
   a working version, I ended up re-implementing this feature from
   scratch as part of a larger code-cleanup, and in order to have
   a single code path for the vectorized and non-vectorized APIs.  I
   subsequently implemented the algorithm by Gladwell to extract
   even more parallelism by evalutating many hypercubes at once.)
*/

template<std::size_t D, typename R>
struct HyperCube {
	Point<D, R> center;
	Point<D, R> half_width;
	R vol;

	HyperCube() = default;
	HyperCube(Point<D, R> center, Point<D, R> half_width) :
			center(center),
			half_width(half_width),
			vol(1) {
		for (std::size_t idx = 0; idx < D; ++idx) {
			vol *= 2 * half_width[idx];
		}
	}
};

template<std::size_t D, typename R>
struct Region {
	HyperCube<D, R> h;
	std::size_t split_dim;
	EstErr<R> est_err;

	Region() = default;
	Region(HyperCube<D, R> h) :
		h(h),
		split_dim(0),
		est_err{ 0, 0 } { }

	// Splits the region into two along split_dim. Returns the other region that
	// gets created.
	Region<D, R> cut() {
		Region<D, R> other(*this);
		h.half_width[split_dim] *= 0.5;
		other.h.half_width[split_dim] *= 0.5;
		h.center[split_dim] -= h.half_width[split_dim];
		other.h.center[split_dim] += other.h.half_width[split_dim];
		h.vol /= 2;
		other.h.vol /= 2;
		return other;
	}

	bool operator<(Region<D, R> const& rhs) const {
		return est_err.err < rhs.est_err.err;
	}
};

template<std::size_t D, typename R, typename F>
struct Rule {
	std::size_t const num_points;
	Rule(std::size_t num_points) : num_points(num_points) { }
	virtual void eval_error(
			F f,
			std::size_t num_regions,
			Region<D, R>* regions) const {
		static_cast<void>(f);
		static_cast<void>(num_regions);
		static_cast<void>(regions);
	};
	virtual ~Rule() = default;
};

// Functions to loop over points in a hypercube. Based on orbitrule.cpp in
// HIntLib-0.0.10.

// ls0 returns the least-significant 0 bit of n (e.g. it returns 0 if the LSB is
// 0, it returns 1 if the 2 LSBs are 01, etc.).
inline std::size_t ls0(std::size_t n) {
#if defined(__GNUC__) && \
	((__GNUC__ == 3 && __GNUC_MINOR__ >= 4) || __GNUC__ >= 4)
	// We can use a GCC built-in for versions >=3.4.
	return __builtin_ctz(~n);
#else
	std::size_t const bits[0xff] = {
		0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4,
		0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5,
		0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4,
		0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 6,
		0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4,
		0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5,
		0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4,
		0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 7,
		0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4,
		0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5,
		0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4,
		0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 6,
		0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4,
		0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5,
		0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4,
		0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 8,
	};
	std::size_t bit = 0;
	while ((n & 0xff) == 0xff) {
		n >>= 8;
		bit += 8;
	}
	return bit + bits[n & 0xff];
#endif
}

// Evaluate the integration points for all 2^n points (+/-r,...+/-r).
//
// A Gray-code ordering is used to minimize the number of coordinate updates in
// p, although this doesn't matter as much now that we are saving all points.
template<std::size_t D, typename R>
void fill_points_R_Rfs(
		Point<D, R>* points_out,
		Point<D, R> center,
		Point<D, R> width) {
	std::size_t signs = 0;
	Point<D, R> point = center;
	for (std::size_t d = 0; d < D; ++d) {
		point[d] += width[d];
	}
	for (std::size_t n = 0; ; ++n) {
		*points_out = point; ++points_out;
		std::size_t d = ls0(n);
		if (d >= D) {
			break;
		}
		std::size_t mask = 1 << d;
		signs ^= mask;
		int sign = (signs & mask) ? -1 : +1;
		point[d] = center[d] + sign * width[d];
	}
}

template<std::size_t D, typename R>
void fill_points_RR0_0fs(
		Point<D, R>* points_out,
		Point<D, R> center,
		Point<D, R> width) {
	Point<D, R> point = center;
	for (std::size_t d1 = 0; d1 < D - 1; ++d1) {
		point[d1] = center[d1] - width[d1];
		for (std::size_t d2 = d1 + 1; d2 < D; ++d2) {
			point[d2] = center[d2] - width[d2];
			*points_out = point; ++points_out;
			point[d1] = center[d1] + width[d1];
			*points_out = point; ++points_out;
			point[d2] = center[d2] + width[d2];
			*points_out = point; ++points_out;
			point[d1] = center[d1] - width[d1];
			*points_out = point; ++points_out;
			point[d2] = center[d2];
		}
		point[d1] = center[d1];
	}
}

template<std::size_t D, typename R>
void fill_points_R0_0fs4d(
		Point<D, R>* points_out,
		Point<D, R> center,
		Point<D, R> width1, Point<D, R> width2) {
	Point<D, R> point = center;
	*points_out = point; ++points_out;
	for (std::size_t d = 0; d < D; ++d) {
		point[d] = center[d] - width1[d];
		*points_out = point; ++points_out;
		point[d] = center[d] + width1[d];
		*points_out = point; ++points_out;
		point[d] = center[d] - width2[d];
		*points_out = point; ++points_out;
		point[d] = center[d] + width2[d];
		*points_out = point; ++points_out;
		point[d] = center[d];
	}
}

template<std::size_t D>
constexpr std::size_t num_0_0() {
	return 1;
}
template<std::size_t D>
constexpr std::size_t num_R0_0fs() {
	return 2 * D;
}
template<std::size_t D>
constexpr std::size_t num_RR0_0fs() {
	return 2 * D * (D - 1);
}
template<std::size_t D>
constexpr std::size_t num_R_Rfs() {
	return 1 << D;
}

// Based on rule75genzmalik.cpp in HIntLib-0.0.10: An embedded cubature rule of
// degree 7 (embedded rule degree 5) due to A. C. Genz and A. A. Malik.  See:
// 
// A. C. Genz and A. A. Malik, "An imbedded [sic] family of fully symmetric
// numerical integration rules," SIAM J. Numer. Anal. 20 (3), 580-588 (1983).
template<std::size_t D, typename R, typename F>
struct Rule75GenzMalik : public Rule<D, R, F> {
	static_assert(
		D >= 2,
		"Rule75GenzMalik only supports dim 2 or larger");
	static_assert(
		D < sizeof(std::size_t) * 8,
		"Rule75GenzMalik is restricted to dimensions smaller than 8*sizeof(size_t)");
	Rule75GenzMalik() : Rule<D, R, F>(
		num_0_0<D>()
		+ 2 * num_R0_0fs<D>()
		+ num_RR0_0fs<D>()
		+ num_R_Rfs<D>()) { }

	void eval_error(
			F f,
			std::size_t num_regions,
			Region<D, R>* regions) const override {
		R const lambda2 = 0.35856858280031809199064515390793749545406372969943094277622494703446518825470728992749201941355065835659927094287145157207750929L;
		R const lambda4 = 0.94868329805051379959966806332981556011586654179756504805725145583777833159177146640327443251379008855620418524585201654456465681L;
		R const lambda5 = 0.68824720161168529772162873429362352512689535661564920056163328497256168340049864510360407723978071477457767704709592846782905149L;
		R const weight1 = (12824.L - (9120.L - 400.L * D) * D) / 19683.L;
		R const weight2 = 980.L / 6561.L;
		R const weight3 = (1820.L - 400.L * D) / 19683.L;
		R const weight4 = 200.L / 19683.L;
		R const weight5 = 6859.L / 19683.L / R(1 << D);
		R const weightE1 = (729.L - 50.L * (19.L - D) * D) / 729.L;
		R const weightE2 = 245.L / 486.L;
		R const weightE3 = (265.L - 100.L * D) / 1458.L;
		R const weightE4 = 25.L / 729.L;
		R const ratio = (lambda2 * lambda2) / (lambda4 * lambda4);
		// Need an array big enough to store all of the points and vals.
		std::vector<Point<D, R> > points(num_regions * this->num_points);
		std::vector<R> vals(num_regions * this->num_points);

		for (std::size_t idx = 0; idx < num_regions; ++idx) {
			Point<D, R>* points_R0_0fs4d = &points[idx * this->num_points];
			Point<D, R>* points_RR0_0fs = points_R0_0fs4d + num_0_0<D>() + 2 * num_R0_0fs<D>();
			Point<D, R>* points_R_Rfs = points_RR0_0fs + num_RR0_0fs<D>();

			Point<D, R> center = regions[idx].h.center;
			Point<D, R> half_width = regions[idx].h.half_width;
			Point<D, R> half_width_lambda2 = half_width;
			Point<D, R> half_width_lambda4 = half_width;
			Point<D, R> half_width_lambda5 = half_width;
			for (std::size_t d = 0; d < D; ++d) {
				half_width_lambda2[d] *= lambda2;
				half_width_lambda4[d] *= lambda4;
				half_width_lambda5[d] *= lambda5;
			}

			// Fill in the points to the array.
			fill_points_R0_0fs4d(points_R0_0fs4d, center, half_width_lambda2, half_width_lambda4);
			fill_points_RR0_0fs(points_RR0_0fs, center, half_width_lambda4);
			fill_points_R_Rfs(points_R_Rfs, center, half_width_lambda5);
		}

		// Evaluate the integrand at all points.
		f(this->num_points * num_regions, points.data(), vals.data());

		// Calculate integral and error.
		std::vector<Point<D, R> > diff = points;
		diff.resize(num_regions);
		for (std::size_t idx = 0; idx < num_regions; ++idx) {
			diff[idx] = Point<D, R>{{}};
		}
		for (std::size_t idx = 0; idx < num_regions; ++idx) {
			R* vals_sl = &vals.data()[idx * this->num_points];
			R sum2 = 0, sum3 = 0, sum4 = 0, sum5 = 0;
			R val0 = vals_sl[0];
			std::size_t d0 = 1;
			for (std::size_t d = 0; d < D; ++d) {
				R v0 = vals_sl[d0 + 4 * d + 0];
				R v1 = vals_sl[d0 + 4 * d + 1];
				R v2 = vals_sl[d0 + 4 * d + 2];
				R v3 = vals_sl[d0 + 4 * d + 3];
				sum2 += v0 + v1;
				sum3 += v2 + v3;
				diff[idx][d] += std::fabs(v0 + v1 - 2 * val0 - ratio * (v2 + v3 - 2 * val0));
			}
			d0 += 4 * D;
			for (std::size_t d = 0; d < num_RR0_0fs<D>(); ++d) {
				sum4 += vals_sl[d0 + d];
			}
			d0 += num_RR0_0fs<D>();
			for (std::size_t d = 0; d < num_R_Rfs<D>(); ++d) {
				sum5 += vals_sl[d0 + d];
			}
			d0 += num_R_Rfs<D>();

			// Calculate 5th and 7th order results.
			R result = regions[idx].h.vol * (weight1 * val0 + weight2 * sum2 + weight3 * sum3 + weight4 * sum4 + weight5 * sum5);
			R res5th = regions[idx].h.vol * (weightE1 * val0 + weightE2 * sum2 + weightE3 * sum3 + weightE4 * sum4);
			regions[idx].est_err = { result, std::fabs(res5th - result) };
		}

		// Figure out dimension to split.
		for (std::size_t idx = 0; idx < num_regions; ++idx) {
			R max_diff = 0;
			std::size_t dim_max_diff = 0;
			for (std::size_t d = 0; d < D; ++d) {
				R next_diff = std::fabs(diff[idx][d]);
				if (next_diff > max_diff) {
					max_diff = next_diff;
					dim_max_diff = d;
				}
			}
			regions[idx].split_dim = dim_max_diff;
		}
	}
};

template<typename R, typename F>
struct Rule15Gauss : public Rule<1, R, F> {
	Rule15Gauss() : Rule<1, R, F>(15) { }
	void eval_error(
			F f,
			std::size_t num_regions,
			Region<1, R>* regions) const override {
		std::size_t const n = 8;
		// Points for the 15-point Konrod rule.
		R const xgk[8] = {
			0.9914553711208126392068546975263285L,
			0.9491079123427585245261896840478513L,
			0.8648644233597690727897127886409262L,
			0.7415311855993944398638647732807884L,
			0.5860872354676911302941448382587296L,
			0.4058451513773971669066064120769615L,
			0.2077849550078984676006894037732449L,
			0.0000000000000000000000000000000000L,
		};
		// Weights for the 15-point Konrod rule.
		R const wgk[8] = {
			2.293532201052922496373200805896959e-2L,
			6.309209262997855329070066318920429e-2L,
			1.047900103222501838398763225415180e-1L,
			1.406532597155259187451895905102379e-1L,
			1.690047266392679028265834265985503e-1L,
			1.903505780647854099132564024210137e-1L,
			2.044329400752988924141619992346491e-1L,
			2.094821410847278280129991748917143e-1L,
		};
		// Weights of the 7-point Gauss rule.
		R const wg[4] = {
			0.1294849661688696932706114326790820L,
			0.2797053914892766679014677714237796L,
			0.3818300505051189449503697754889751L,
			0.4179591836734693877551020408163265L,
		};

		std::vector<Point<1, R>> points(this->num_points * num_regions);
		std::vector<R> vals(this->num_points * num_regions);

		// Find the Gauss-Konrod points.
		for (std::size_t idx = 0; idx < num_regions; ++idx) {
			R center = regions[idx].h.center[0];
			R half_width = regions[idx].h.half_width[0];
			points[idx * this->num_points + 0] = { center };
			std::size_t j0 = 1;
			for (std::size_t j = 0; j < (n - 1) / 2; ++j) {
				std::size_t j2 = 2 * j + 1;
				R width = half_width * xgk[j2];
				points[idx * this->num_points + j0 + 2 * j + 0] = { center - width };
				points[idx * this->num_points + j0 + 2 * j + 1] = { center + width };
			}
			j0 += 2 * ((n - 1) / 2);
			for (std::size_t j = 0; j < n / 2; ++j) {
				std::size_t j2 = 2 * j;
				R width = half_width * xgk[j2];
				points[idx * this->num_points + j0 + 2 * j + 0] = { center - width };
				points[idx * this->num_points + j0 + 2 * j + 1] = { center + width };
			}
			j0 += 2 * (n / 2);
			regions[idx].split_dim = 0;
		}

		// Evaluate the function at the points.
		f(this->num_points * num_regions, points.data(), vals.data());

		// Calculate integral and error. This comes from the GSL.
		for (std::size_t idx = 0; idx < num_regions; ++idx) {
			R* vals_sl = &vals.data()[idx * this->num_points];

			R half_width = regions[idx].h.half_width[0];

			R result_gauss = vals_sl[0] * wg[n / 2 - 1];
			R result_konrod = vals_sl[0] * wgk[n - 1];
			R result_abs = std::fabs(result_konrod);
			std::size_t j0 = 1;
			for (std::size_t j = 0; j < (n - 1) / 2; ++j) {
				std::size_t j2 = 2 * j + 1;
				R v1 = vals_sl[j0 + 2 * j + 0];
				R v2 = vals_sl[j0 + 2 * j + 1];
				result_gauss += wg[j] * (v1 + v2);
				result_konrod += wgk[j2] * (v1 + v2);
				result_abs += wgk[j2] * (std::fabs(v1) + std::fabs(v2));
			}
			j0 += 2 * ((n - 1) / 2);
			for (std::size_t j = 0; j < n / 2; ++j) {
				std::size_t j2 = 2 * j;
				R v1 = vals_sl[j0 + 2 * j + 0];
				R v2 = vals_sl[j0 + 2 * j + 1];
				result_konrod += wgk[j2] * (v1 + v2);
				result_abs += wgk[j2] * (std::fabs(v1) + std::fabs(v2));
			}
			j0 += 2 * (n / 2);

			regions[idx].est_err.val = result_konrod * half_width;

			// Error estimate.
			R mean = 0.5 * result_konrod;
			R result_asc = wgk[n - 1] * std::fabs(vals_sl[0] - mean);
			j0 = 1;
			for (std::size_t j = 0; j < (n - 1) / 2; ++j) {
				std::size_t j2 = 2 * j + 1;
				R v1 = vals_sl[j0 + 2 * j + 0];
				R v2 = vals_sl[j0 + 2 * j + 1];
				result_asc += wgk[j2] * (std::fabs(v1 - mean) + std::fabs(v2 - mean));
			}
			j0 += 2 * ((n - 1) / 2);
			for (std::size_t j = 0; j < n / 2; ++j) {
				std::size_t j2 = 2 * j;
				R v1 = vals_sl[j0 + 2 * j + 0];
				R v2 = vals_sl[j0 + 2 * j + 1];
				result_asc += wgk[j2] * (std::fabs(v1 - mean) + std::fabs(v2 - mean));
			}
			j0 += 2 * (n / 2);
			R err = std::fabs(result_konrod - result_gauss) * half_width;
			result_abs *= half_width;
			result_asc *= half_width;
			if (result_asc != 0 && err != 0) {
				R scale = std::pow((200 * err / result_asc), 1.5);
				err = (scale < 1) ? result_asc * scale : result_asc;
			}
			if (result_abs > std::numeric_limits<R>::min() / (50 * std::numeric_limits<R>::epsilon())) {
				R min_err = 50 * std::numeric_limits<R>::epsilon() * result_abs;
				if (min_err > err) {
					err = min_err;
				}
			}
			regions[idx].est_err.err = err;
		}
	}
};

template<typename T>
T sqr(T x) {
	return x * x;
}

template<typename R>
bool converged(
		EstErr<R> const& est_err,
		R req_abs_err, R req_rel_err) {
	return est_err.err <= req_abs_err
		|| est_err.err <= std::fabs(est_err.val) * req_rel_err;
}

// Adaptive integration.
template<std::size_t D, typename R, typename F>
EstErr<R> rule_cubature(
		Rule<D, R, F> const& r,
		F f,
		HyperCube<D, R> const& h,
		std::size_t max_eval,
		R req_abs_err, R req_rel_err,
		bool parallel) {
	std::size_t num_eval = 0;
	EstErr<R> total_est_err { 0, 0 };

	Region<D, R> region_init(h);
	r.eval_error(f, 1, &region_init);
	num_eval += r.num_points;

	std::vector<Region<D, R> > regions;
	regions.push_back(region_init);
	std::make_heap(regions.begin(), regions.end());
	total_est_err.val += region_init.est_err.val;
	total_est_err.err += region_init.est_err.err;

	while (num_eval < max_eval || !max_eval) {
		if (converged(total_est_err, req_abs_err, req_rel_err)) {
			break;
		}
		if (parallel) {
			// Adapted from I. Gladwell, "Vectorization of one dimensional
			// quadrature codes," pp. 230--238 in _Numerical Integration. Recent
			// Developments, Software and Applications_, G. Fairweather and P.
			// M. Keast, eds., NATO ASI Series C203, Dordrecht (1987), as
			// described in J. M. Bull and T. L. Freeman, "Parallel Globally
			// Adaptive Algorithms for Multi-dimensional Integration,"
			// http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.42.6638(1994). 
			// 
			// Basically, this evaluates in one shot all regions that *must* be
			// evaluated in order to reduce the error to the requested bound:
			// the minimum set of largest-error regions whose errors push the
			// total error over the bound.
			// 
			// [Note: Bull and Freeman claim that the Gladwell approach is
			// intrinsically inefficent because it "requires sorting", and
			// propose an alternative algorithm that "only" requires three
			// passes over the entire set of regions.  Apparently, they didn't
			// realize that one could use a heap data structure, in which case
			// the time to pop K biggest-error regions out of D is only
			// O(K log D), much better than the O(D) cost of the Bull and
			// Freeman algorithm if K << D, and it is also much simpler.]
			std::vector<Region<D, R> > next_regions;
			do {
				std::pop_heap(regions.begin(), regions.end());
				next_regions.push_back(regions.back());
				EstErr<R> prev_est_err = regions.back().est_err;
				regions.pop_back();
				total_est_err.val -= prev_est_err.val;
				total_est_err.err -= prev_est_err.err;
				next_regions.push_back(next_regions.back().cut());
				num_eval += 2 * r.num_points;
				if (converged(total_est_err, req_abs_err, req_rel_err)) {
					// Other regions have small error.
					break;
				}
			} while(!regions.empty() && (num_eval < max_eval || !max_eval));
			r.eval_error(f, next_regions.size(), next_regions.data());
			for (Region<D, R> const& region : next_regions) {
				regions.push_back(region);
				std::push_heap(regions.begin(), regions.end());
				total_est_err.val += region.est_err.val;
				total_est_err.err += region.est_err.err;
			}
			next_regions.clear();
		} else {
			Region<D, R> next_regions[2];
			std::pop_heap(regions.begin(), regions.end());
			next_regions[0] = regions.back();
			EstErr<R> prev_est_err = regions.back().est_err;
			regions.pop_back();
			total_est_err.val -= prev_est_err.val;
			total_est_err.err -= prev_est_err.err;

			next_regions[1] = next_regions[0].cut();
			r.eval_error(f, 2, &next_regions[0]);

			regions.push_back(next_regions[0]);
			std::push_heap(regions.begin(), regions.end());
			regions.push_back(next_regions[1]);
			std::push_heap(regions.begin(), regions.end());
			for (std::size_t idx = 0; idx < 2; ++idx) {
				total_est_err.val += next_regions[idx].est_err.val;
				total_est_err.err += next_regions[idx].est_err.err;
			}
			num_eval += 2 * r.num_points;
		}
	}

	// Re-sum integrals and errors.
	EstErr<R> result { 0, 0 };
	for (Region<D, R> const& region : regions) {
		result.val += region.est_err.val;
		result.err += region.est_err.err;
	}
	return result;
}

template<std::size_t D, typename R, typename F>
struct CubatureBase {
	static EstErr<R> cubature_v_base(
			F f,
			Point<D, R> xmin, Point<D, R> xmax,
			std::size_t max_eval,
			R req_abs_err, R req_rel_err,
			bool parallel) {
		Point<D, R> center;
		Point<D, R> half_width;
		for (std::size_t d = 0; d < D; ++d) {
			center[d] = 0.5 * (xmax[d] + xmin[d]);
			half_width[d] = 0.5 * (xmax[d] - xmin[d]);
		}
		HyperCube<D, R> h(center, half_width);
		return rule_cubature(
			Rule75GenzMalik<D, R, F>(),
			f, h, max_eval, req_abs_err, req_rel_err, parallel);
	}
};

template<typename R, typename F>
struct CubatureBase<0, R, F> {
	static EstErr<R> cubature_v_base(
			F f,
			Point<0, R> xmin, Point<0, R> xmax,
			std::size_t max_eval,
			R req_abs_err, R req_rel_err,
			bool parallel) {
		// Trivial integration.
		static_cast<void>(xmax);
		static_cast<void>(max_eval);
		static_cast<void>(req_abs_err);
		static_cast<void>(req_rel_err);
		static_cast<void>(parallel);
		R result;
		f(1, &xmin, &result);
		return EstErr<R>{ result, 0 };
	}
};

template<typename R, typename F>
struct CubatureBase<1, R, F> {
	static EstErr<R> cubature_v_base(
			F f,
			Point<1, R> xmin, Point<1, R> xmax,
			std::size_t max_eval,
			R req_abs_err, R req_rel_err,
			bool parallel) {
		HyperCube<1, R> h(
			{ 0.5 * (xmax[0] + xmin[0]) },
			{ 0.5 * (xmax[0] - xmin[0]) });
		return rule_cubature(
			Rule15Gauss<R, F>(),
			f, h, max_eval, req_abs_err, req_rel_err, parallel);
	}
};

template<std::size_t D, typename R, typename F>
void check_template_params() {
	static_assert(
		std::is_floating_point<R>::value,
		"Template parameter R must be a flotaing point type");
}

}

template<std::size_t D, typename R, typename F>
EstErr<R> cubature(
		F f,
		Point<D, R> xmin, Point<D, R> xmax,
		std::size_t max_eval,
		R req_abs_err, R req_rel_err) {
	internal::check_template_params<D, R, F>();
	auto fp = [&](std::size_t n, Point<D, R> const* points, R* vals) {
		for (std::size_t idx = 0; idx < n; ++idx) {
			vals[idx] = f(points[idx]);
		}
	};
	return internal::CubatureBase<D, R, decltype(fp)>::cubature_v_base(
		fp, xmin, xmax, max_eval, req_abs_err, req_rel_err, false);
}

template<std::size_t D, typename R, typename FV>
EstErr<R> cubature_v(
		FV f,
		Point<D, R> xmin, Point<D, R> xmax,
		std::size_t max_eval,
		R req_abs_err, R req_rel_err) {
	internal::check_template_params<D, R, FV>();
	return internal::CubatureBase<D, R, FV>::cubature_v_base(
		f, xmin, xmax, max_eval, req_abs_err, req_rel_err, true);
}

template<typename R, typename F>
EstErr<R> cubature(
		F f,
		R xmin, R xmax,
		std::size_t max_eval,
		R req_abs_err, R req_rel_err) {
	auto fp = [&](std::size_t n, cubature::Point<1, R> const* points, R* vals) {
		for (std::size_t idx = 0; idx < n; ++idx) {
			vals[idx] = f(points[idx][0]);
		}
	};
	internal::check_template_params<1, R, decltype(fp)>();
	return internal::CubatureBase<1, R, decltype(fp)>::cubature_v_base(
		fp, { xmin }, { xmax }, max_eval, req_abs_err, req_rel_err, false);
}

template<typename R, typename FV>
EstErr<R> cubature_v(
		FV f,
		R xmin, R xmax,
		std::size_t max_eval,
		R req_abs_err, R req_rel_err) {
	static_assert(
		sizeof(cubature::Point<1, R>) == sizeof(R),
		"cubature::Point must be directly convertable to type R");
	static_assert(
		*static_cast<R const*>(&cubature::Point<1, R>{ 0.12345 })
			== 0.12345,
		"cubature::Point must be directly convertable to type R");
	auto fp = [&](std::size_t n, cubature::Point<1, R> const* points, R* vals) {
		return f(n, static_cast<R const*>(points), vals);
	};
	internal::check_template_params<1, R, decltype(fp)>();
	return internal::CubatureBase<1, R, decltype(fp)>::cubature_v_base(
		fp, { xmin }, { xmax }, max_eval, req_abs_err, req_rel_err, true);
}

}

#endif

