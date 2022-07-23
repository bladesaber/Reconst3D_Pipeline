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
#include "string.h"

#include "types_test.h"

G2O_USE_OPTIMIZATION_LIBRARY(eigen);
G2O_USE_OPTIMIZATION_LIBRARY(dense);

namespace py = pybind11;
using namespace pybind11::literals;

template<typename T1, typename T2>
void EdgeSetVertex(T1& edge, int id, T2& vertex){
    edge.setVertex(id, &vertex);
//    std::cout<<"Edge add vertex success"<<std::endl;
}

template<typename T>
void SetAlgorithm(T& optimizer, std::string& tag, g2o::OptimizationAlgorithmProperty& solverProperty){
    g2o::OptimizationAlgorithm* algo = g2o::OptimizationAlgorithmFactory::instance()->construct(tag,solverProperty);
    optimizer.setAlgorithm(algo);
}

template<typename T>
void PrintfGraphInfo(T& optimizer){
    int edge_count = (optimizer.edges()).size();
    int vertex_count = (optimizer.vertices()).size();
    std::cout<<"Edge Num: "<<edge_count<<std::endl;
    std::cout<<"Vertex Num: "<<vertex_count<<std::endl;
}

template<typename T1>
EdgePointOnCurve OptAddEdge(T1& opt, VertexParams& vertex){
    // todo why ???
    EdgePointOnCurve* e = new EdgePointOnCurve;

    e->setVertex(0, &vertex);
    bool stat = opt.addEdge(e);

    std::cout<< (e->vertices()).size()<<" : "<<stat<<std::endl;

    return *e;
}

void declareOptimizerTypes(py::module &m) {

    py::class_<g2o::OptimizableGraph::Vertex>(m, "OptimizableGraph_Vertex");
    py::class_<g2o::OptimizableGraph::Edge>(m, "OptimizableGraph_Edge");
    py::class_<g2o::HyperGraph::Vertex>(m, "HyperGraph_Vertex");
    py::class_<g2o::HyperGraph::Edge>(m, "HyperGraph_Edge");
    py::class_<g2o::HyperGraph::HyperGraphElement>(m, "HyperGraphElement");

    py::class_<g2o::SparseOptimizer>(m, "SparseOptimizer")
            .def(py::init<>())
            .def("set_verbose", &g2o::SparseOptimizer::setVerbose, "verbose"_a)
            .def("optimize", &g2o::SparseOptimizer::optimize, "iterations"_a, "online"_a=false)
//            .def("initializeOptimization", static_cast<bool (g2o::SparseOptimizer::*)(g2o::HyperGraph::EdgeSet&)>(&g2o::SparseOptimizer::initializeOptimization))
//            .def("initializeOptimization", static_cast<bool (g2o::SparseOptimizer::*)(g2o::HyperGraph::VertexSet&, int)>(&g2o::SparseOptimizer::initializeOptimization))
            .def("initializeOptimization", static_cast<bool (g2o::SparseOptimizer::*)(int)>(&g2o::SparseOptimizer::initializeOptimization), "level"_a=0)
            .def("addVertex", static_cast<bool (g2o::SparseOptimizer::*)(g2o::HyperGraph::Vertex*)>(&g2o::SparseOptimizer::addVertex))
            .def("addVertex", static_cast<bool (g2o::SparseOptimizer::*)(g2o::OptimizableGraph::Vertex*)>(&g2o::SparseOptimizer::addVertex))
            .def("addEdge", static_cast<bool (g2o::SparseOptimizer::*)(g2o::OptimizableGraph::Edge*)>(&g2o::SparseOptimizer::addEdge))
            .def("addEdge", static_cast<bool (g2o::SparseOptimizer::*)(g2o::HyperGraph::Edge*)>(&g2o::SparseOptimizer::addEdge));

    m.def("EdgeSetVertex", &EdgeSetVertex<g2o::OptimizableGraph::Edge, g2o::OptimizableGraph::Vertex>);

    py::class_<g2o::OptimizationAlgorithmProperty>(m, "OptimizationAlgorithmProperty")
            .def(py::init<>());

    m.def("SetAlgorithm", &SetAlgorithm<g2o::SparseOptimizer>);
    m.def("PrintfGraphInfo", &PrintfGraphInfo<g2o::SparseOptimizer>);

    m.def("OptAddEdge", &OptAddEdge<g2o::SparseOptimizer>);

}

#endif //SLAM_PY_TYPES_OPTIMIZER_H
