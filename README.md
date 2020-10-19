Cubature-cpp
------------

This is a C++ port of the [cubature](https://github.com/stevengj/cubature)
library, adapted for use in the [`sidis`](https://github.com/duanebyer/sidis)
event generator. Currently, this port has several limitations compared to the
original library:

* Only scalar integrands are supported (no vector integrands).
* Only the h-adaptive algorithm is supported.
* The dimension of the integrated region must be selected at compile time.

It also has some new features:

* Support for C++ closures and functor objects.
* User-specified real number types.

For more information, visit the original
[cubature](https://github.com/stevengj/cubature) project.

Build
-----

The library is provided in a header-only form, so you can place `cubature.hpp`
and `cubature.ipp` in your include directory to start using it. Alternatively,
build and install using CMake.

Usage
-----

As a simple example:

```cpp
#include <cubature.hpp>

// ...
cubature::ErrEst<double> res = cubature::cubature(
	[&](cubature::Point<3, double> x) {
		return std::sin(x[0]) * std::cos(x[1]) * x[2] * x[2];
	},
	cubature::Point<3, double>{ xmin, ymin, zmin },
	cubature::Point<3, double>{ xmax, ymax, zmax },
	max_eval, req_abs_err, req_rel_err);
std::cout << "Integral: " << res.val << std::endl;
std::cout << "Error: " << res.err << std::endl;
// ...
```

A vectorized interface is also supported. To use it, call `cubature_v` with a
functor of the signature:

```cpp
functor(std::size_t num_points, cubature::Point3<N, R>* points, R* vals);
```

