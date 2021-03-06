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

#ifndef CUBATURE_HPP
#define CUBATURE_HPP

#include <array>
#include <cstddef>

namespace cubature {

template<std::size_t D, typename R>
using Point = std::array<R, D>;

template<typename R>
struct EstErr {
	R val;
	R err;
};

// Integrate the function f from xmin[dim] to xmax[dim], with at most maxEval
// function evaluations (0 for no limit), until the given absolute or relative
// error is achieved.  val returns the integral, and err returns the estimate
// for the absolute error in val; both of these are arrays of length fdim, the
// dimension of the vector integrand f(x). The return value of the function is 0
// on success and non-zero if there  was an error.

// Adapative integration by partitioning the integration domain ("h-adaptive")
// and using the same fixed-degree quadrature in each subdomain, recursively,
// until convergence is achieved.
template<std::size_t D, typename R, typename F>
EstErr<R> cubature(
	F f,
	Point<D, R> xmin, Point<D, R> xmax,
	std::size_t max_eval,
	R req_abs_err, R req_rel_err);

// As cubature, but with a vectorized integrand.
template<std::size_t D, typename R, typename FV>
EstErr<R> cubature_v(
	FV f,
	Point<D, R> xmin, Point<D, R> xmax,
	std::size_t max_eval,
	R req_abs_err, R req_rel_err);

// One-dimensional version that doesn't require wrapping in a Point.
template<typename R, typename F>
EstErr<R> cubature(
	F f,
	R xmin, R xmax,
	std::size_t max_eval,
	R req_abs_err, R req_rel_err);

template<typename R, typename F>
EstErr<R> cubature_v(
	F f,
	R xmin, R xmax,
	std::size_t max_eval,
	R req_abs_err, R req_rel_err);

}

#include "cubature.ipp"

#endif

