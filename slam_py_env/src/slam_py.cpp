//
// Created by quan on 2022/7/20.
//

#include <pybind11/pybind11.h>

//#include "types_eigen.h"
#include "types_test.h"

namespace py = pybind11;

PYBIND11_MODULE(slam_py, m) {
    m.attr("package_name") = "slam_py_package";

    py::class_<Test>(m, "Test")
            .def(py::init<>())
            .def("add", &Test::add);

    py::class_<VertexParams>(m, "VertexParams")
            .def(py::init<>())
            .def("setId", &VertexParams::setId)
            .def("setEstimate", &VertexParams::setEstimate);
    py::class_<EdgePointOnCurve>(m, "EdgePointOnCurve")
            .def(py::init<>())
            .def("setId", &EdgePointOnCurve::setId)
            .def("setVertex", &EdgePointOnCurve::setVertex)
            .def("setMeasurement", &EdgePointOnCurve::setMeasurement)
            .def("setInformation", &EdgePointOnCurve::setInformation);

}