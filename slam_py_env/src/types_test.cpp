//
// Created by quan on 2022/7/20.
//

#include "types_test.h"

int Test::add(int x, int y) {
    return x+y;
}

bool VertexParams::read(std::istream &) {
    return false;
}

bool VertexParams::write(std::ostream &) const {
    return false;
}

void VertexParams::oplusImpl(const double *update) {
    Eigen::Vector3d::ConstMapType v(update);
    _estimate += v;
}

bool EdgePointOnCurve::read(std::istream &) {
    return false;
}

bool EdgePointOnCurve::write(std::ostream &) const {
    return false;
}

template<typename T>
bool EdgePointOnCurve::operator()(const T *params, T *error) const {
    const T &a = params[0];
    const T &b = params[1];
    const T &lambda = params[2];
    T fval = a * exp(-lambda * T(measurement()(0))) + b;
    error[0] = fval - measurement()(1);
    return true;
}

// C++规范，如果有成对.cpp/.h，所有的非类函数不可放在.h，否则产生多重定义
void declareTestTypes(py::module &m) {
    py::class_<Test>(m, "Test")
            .def(py::init<>())
            .def("add", &Test::add);

    py::class_<VertexParams, g2o::OptimizableGraph::Vertex, g2o::HyperGraph::Vertex>(m, "VertexParams")
            .def(py::init<>())
            .def("setId", &VertexParams::setId);
//            .def("estimate", &VertexParams::estimate);
//            .def("setEstimate", &VertexParams::setEstimate);

    py::class_<EdgePointOnCurve, g2o::OptimizableGraph::Edge, g2o::HyperGraph::Edge>(m, "EdgePointOnCurve")
            .def(py::init<>())
            .def("setId", &EdgePointOnCurve::setId);
//            .def("setVertex", &EdgePointOnCurve::setVertex, "i"_a, "v"_a);
//            .def("setMeasurement", &EdgePointOnCurve::setMeasurement)
//            .def("setInformation", &EdgePointOnCurve::setInformation);

    py::class_<DataFitingOlver::EdgeReturn>(m, "DataFitingOlver_EdgeReturn")
            .def_readonly("edge", &DataFitingOlver::EdgeReturn::edge)
            .def_readonly("stat", &DataFitingOlver::EdgeReturn::stat);
    py::class_<DataFitingOlver>(m, "DataFitingOlver")
            .def(py::init<>())
            .def("setVerbose", &DataFitingOlver::setVerbose, "verbos"_a)
            .def("setAlgorithm", &DataFitingOlver::setAlgorithm)
            .def("addVertex", &DataFitingOlver::addVertex)
            .def("addEdge", static_cast<DataFitingOlver::EdgeReturn* (DataFitingOlver::*)(VertexParams&, int)>(&DataFitingOlver::addEdge), "vertex"_a, "id"_a)
            .def("addEdge", static_cast<bool (DataFitingOlver::*)(EdgePointOnCurve&)>(&DataFitingOlver::addEdge), "edge"_a)
            .def("initializeOptimization", &DataFitingOlver::initializeOptimization, "level"_a=0)
            .def("optimize", &DataFitingOlver::optimize, "maxIterations"_a, "online"_a= false)
            .def("getVertex", &DataFitingOlver::getVertex)
            .def("getEdge", &DataFitingOlver::getEdge)
            .def("VertexSetEstimate", &DataFitingOlver::VertexSetEstimate<Eigen::Vector3d>, "vertex"_a, "data"_a)
            .def("EdgeSetInformation", &DataFitingOlver::EdgeSetInformation<Eigen::Matrix<double, 1, 1>>)
            .def("EdgeSetMeasurement", &DataFitingOlver::EdgeSetMeasurement<Eigen::Vector2d>)
            .def("VertexGetEstimate", &DataFitingOlver::VertexGetEstimate)
            .def("EdgeSetVertex", &DataFitingOlver::EdgeSetVertex)
            .def_readonly("optimizer", &DataFitingOlver::optimizer);

}