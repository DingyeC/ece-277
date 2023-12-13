/*************************************************************************
/* ECE 277: GPU Programmming 2020
/* Author and Instructer: Cheolhong An
/* Copyright 2020
/* University of California, San Diego
/*************************************************************************/

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

extern void cu_mmul(int* A, int* W, int* C, const int M, const int N, const int K);

namespace py = pybind11;


py::array_t<int> mmul_wrapper(py::array_t<int> a1, py::array_t<int> a2) 
{
	auto buf1 = a1.request();
	auto buf2 = a2.request();

	if (a1.ndim() != 2 || a2.ndim() != 2)
		throw std::runtime_error("Number of dimensions must be two");

	// NxM matrix
	int M = a1.shape()[0];
	int N = a2.shape()[1];
	printf("M=%d, N=%d\n", M, N);

	auto result = py::array(py::buffer_info(
		nullptr,            /* Pointer to data (nullptr -> ask NumPy to allocate!) */
		sizeof(int),     /* Size of one item */
		py::format_descriptor<int>::value, /* Buffer format */
		buf1.ndim,          /* How many dimensions? */
		{ M, N},  /* Number of elements for each dimension */
		{ sizeof(int)*N, sizeof(int) }  /* Strides for each dimension */
	));

	auto buf3 = result.request();

	int* A = (int*)buf1.ptr;
	int* W = (int*)buf2.ptr;
	int* C = (int*)buf3.ptr;

	cu_mmul(A, W, C, M, N, K);

    return result;
}



PYBIND11_MODULE(cu_matrix_mul, m) {
    m.def("mmul", &mmul_wrapper, "Multiply two NumPy arrays");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
