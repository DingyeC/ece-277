/*************************************************************************
/* ECE 277: GPU Programmming 2020
/* Author and Instructer: Cheolhong An
/* Copyright 2020
/* University of California, San Diego
/*************************************************************************/

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

extern void cu_mmac(int* A, int* W, int* C, const int M, const int N, const int K);

namespace py = pybind11;


py::array_t<int> mmac_wrapper(py::array_t<int> a1, py::array_t<int> a2, py::array_t<int> a3) 
{
	auto buf1 = a1.request();
	auto buf2 = a2.request();
	auto buf3 = a3.request();

	if (a1.ndim() != 2 || a2.ndim() != 2 || a3.ndim() != 2)
		throw std::runtime_error("Number of dimensions must be two");

	// NxM matrix
	int M = a3.shape()[0];
	int N = a3.shape()[1];
	int K = a1.shape()[1];
	//printf("M=%d, N=%d, k=%d, k=%d\n", M, N, K, a2.shape()[0]);

	int* A = (int*)buf1.ptr;
	int* W = (int*)buf2.ptr;
	int* C = (int*)buf3.ptr;

	cu_mmac(A, W, C, M, N, K);

    return a3;
}



PYBIND11_MODULE(cu_matrix_mmac, m) {
    m.def("mmac", &mmac_wrapper, "MAC");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
