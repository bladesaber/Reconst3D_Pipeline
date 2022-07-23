//
// Created by quan on 2022/7/20.
//

#ifndef SLAM_PY_TYPES_EIGEN_H
#define SLAM_PY_TYPES_EIGEN_H

#include <pybind11/pybind11.h>
#include "iostream"
#include <g2o/core/eigen_types.h>

#include "pybind11/eigen.h"
#include "pybind11/stl.h"

#include <Eigen/Core>
#include <Eigen/src/Core/DenseCoeffsBase.h>

#include "types_test.h"

namespace py = pybind11;
using namespace pybind11::literals;

void declareEigenTypes(py::module &m) {

    /*
    // if you use pybind11/eigen.h, please do not define the object again
    py::class_<Eigen::Vector3d>(m, "Vector3d")
            .def(py::init<>())
            .def(py::init<const Eigen::Vector3d &>())
            .def(py::init<float &, float &, float &>(), "x"_a, "y"_a, "z"_a)
            .def("x", [](Eigen::Vector3d* v){return v->x();})
            .def("y", [](Eigen::Vector3d* v){return v->y();})
            .def("z", [](Eigen::Vector3d* v){return v->z();})
            .def("setX", [](Eigen::Vector3d* v, float x){v->x()=x;})
            .def("setY", [](Eigen::Vector3d* v, float y){v->y()=y;})
            .def("setZ", [](Eigen::Vector3d* v, float z){v->z()=z;});

    py::class_<Eigen::Vector2d>(m, "Vector2d")
            .def(py::init<>())
            .def(py::init<const Eigen::Vector2d &>())
            .def(py::init<float &, float &>(), "x"_a, "y"_a)
            .def("x", [](Eigen::Vector2d* v){return v->x();})
            .def("y", [](Eigen::Vector2d* v){return v->y();})
            .def("setX", [](Eigen::Vector2d* v, float x){v->x()=x;})
            .def("setY", [](Eigen::Vector2d* v, float y){v->y()=y;});
    */

}

#endif //SLAM_PY_TYPES_EIGEN_H
