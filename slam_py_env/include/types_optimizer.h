//
// Created by quan on 2022/7/21.
//

#ifndef SLAM_PY_TYPES_OPTIMIZER_H
#define SLAM_PY_TYPES_OPTIMIZER_H

#include "iostream"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "g2o/core/sparse_optimizer.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/solvers/eigen//linear_solver_eigen.h"
#include "g2o/core/optimization_algorithm_factory.h"
#include "string.h"

namespace py = pybind11;
using namespace pybind11::literals;

G2O_USE_OPTIMIZATION_LIBRARY(eigen);
G2O_USE_OPTIMIZATION_LIBRARY(dense);

template<typename T>
void PrintfGraphInfo(T& optimizer){
    int edge_count = (optimizer.edges()).size();
    int vertex_count = (optimizer.vertices()).size();
    std::cout<<"[DEBUG-g2o]: Edge Num: "<<edge_count<<std::endl;
    std::cout<<"[DEBUG-g2o]: Vertex Num: "<<vertex_count<<std::endl;
}

void declareOptimizerTypes(py::module &m) {

    py::class_<g2o::OptimizableGraph::Vertex>(m, "OptimizableGraph_Vertex");
    py::class_<g2o::OptimizableGraph::Edge>(m, "OptimizableGraph_Edge");
    py::class_<g2o::HyperGraph::Vertex>(m, "HyperGraph_Vertex");
    py::class_<g2o::HyperGraph::Edge>(m, "HyperGraph_Edge");
    py::class_<g2o::HyperGraph::HyperGraphElement>(m, "HyperGraphElement");

    py::class_<g2o::SparseOptimizer>(m, "SparseOptimizer")
            .def(py::init<>())
            .def("set_verbose", &g2o::SparseOptimizer::setVerbose, "verbose"_a);
//            .def("optimize", &g2o::SparseOptimizer::optimize, "iterations"_a, "online"_a=false)
//            .def("initializeOptimization", static_cast<bool (g2o::SparseOptimizer::*)(g2o::HyperGraph::EdgeSet&)>(&g2o::SparseOptimizer::initializeOptimization))
//            .def("initializeOptimization", static_cast<bool (g2o::SparseOptimizer::*)(g2o::HyperGraph::VertexSet&, int)>(&g2o::SparseOptimizer::initializeOptimization))
//            .def("initializeOptimization", static_cast<bool (g2o::SparseOptimizer::*)(int)>(&g2o::SparseOptimizer::initializeOptimization), "level"_a=0)
//            .def("addVertex", static_cast<bool (g2o::SparseOptimizer::*)(g2o::HyperGraph::Vertex*)>(&g2o::SparseOptimizer::addVertex))
//            .def("addVertex", static_cast<bool (g2o::SparseOptimizer::*)(g2o::OptimizableGraph::Vertex*)>(&g2o::SparseOptimizer::addVertex))
//            .def("addEdge", static_cast<bool (g2o::SparseOptimizer::*)(g2o::OptimizableGraph::Edge*)>(&g2o::SparseOptimizer::addEdge))
//            .def("addEdge", static_cast<bool (g2o::SparseOptimizer::*)(g2o::HyperGraph::Edge*)>(&g2o::SparseOptimizer::addEdge));

    py::class_<g2o::OptimizationAlgorithmProperty>(m, "OptimizationAlgorithmProperty")
            .def(py::init<>());

    m.def("PrintfGraphInfo", &PrintfGraphInfo<g2o::SparseOptimizer>);

}

#endif //SLAM_PY_TYPES_OPTIMIZER_H
