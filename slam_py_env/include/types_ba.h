//
// Created by quan on 2022/7/26.
//

#ifndef SLAM_PY_TYPES_BA_H
#define SLAM_PY_TYPES_BA_H

#include "iostream"
#include "vector"

#include <pybind11/pybind11.h>
#include<pybind11/numpy.h>

#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
#include "g2o/types/sim3/types_seven_dof_expmap.h"

namespace py = pybind11;
using namespace pybind11::literals;

class BASolver{
public:
    g2o::SparseOptimizer optimizer;
    g2o::OptimizationAlgorithmLevenberg* solver;

    BASolver(){
        solver = new g2o::OptimizationAlgorithmLevenberg(
                g2o::make_unique<g2o::BlockSolver_6_3>(
                        g2o::make_unique<g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>>())
                        );
        optimizer.setAlgorithm(solver);
    };

    void setVerbose(bool verbos){
        this->optimizer.setVerbose(verbos);
    };

    bool addPose(g2o::VertexSE3Expmap& pose){
        return this->optimizer.addVertex(&pose);
    }

    bool addPoint(g2o::VertexPointXYZ& point){
        return this->optimizer.addVertex(&point);
    }

    bool addEdge(g2o::EdgeSE3ProjectXYZ& edge){
        return this->optimizer.addEdge(&edge);
    }

    struct EdgeReturn{
    public:
        g2o::EdgeSE3ProjectXYZ* edge;
        bool stat;
    };
    EdgeReturn* addEdge(int PointId, int PoseId,
                        float fx, float fy, float cx, float cy
                        ){
        g2o::EdgeSE3ProjectXYZ* edge = new g2o::EdgeSE3ProjectXYZ();
        edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(PointId)));
        edge->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(PoseId)));
        edge->fx = fx;
        edge->fy = fy;
        edge->cx = cx;
        edge->cy = cy;

        bool stat = this->optimizer.addEdge(edge);
        EdgeReturn* res = new EdgeReturn();
        res->edge = edge;
        res->stat = stat;
        return res;
    }

    EdgeReturn* addEdge(g2o::VertexPointXYZ& point, g2o::VertexSE3Expmap& pose,
                        float fx, float fy, float cx, float cy
    ){
        g2o::EdgeSE3ProjectXYZ* edge = new g2o::EdgeSE3ProjectXYZ();
        edge->setVertex(0, &point);
        edge->setVertex(1, &pose);
        edge->fx = fx;
        edge->fy = fy;
        edge->cx = cx;
        edge->cy = cy;

        bool stat = this->optimizer.addEdge(edge);
        EdgeReturn* res = new EdgeReturn();
        res->edge = edge;
        res->stat = stat;
        return res;
    }

    void PointSetEstimate(g2o::VertexPointXYZ& point, Eigen::Matrix<double,3,1>& data){
        point.setEstimate(data);
    }

    void PoseSetEstimate(g2o::VertexSE3Expmap& pose, g2o::SE3Quat& data){
        pose.setEstimate(data);
    }

    void EdgeSetInformation(g2o::EdgeSE3ProjectXYZ& edge, Eigen::Matrix2d& info){
        edge.setInformation(info);
    }

    void EdgeSetMeasurement(g2o::EdgeSE3ProjectXYZ& edge, Eigen::Matrix2d meas){
        edge.setMeasurement(meas);
    }

    void EdgeSetVertex(g2o::EdgeSE3ProjectXYZ& edge, int id, g2o::VertexSE3Expmap& pose){
        edge.setVertex(id, &pose);
    }

    void EdgeSetVertex(g2o::EdgeSE3ProjectXYZ& edge, int id, g2o::VertexPointXYZ& point){
        edge.setVertex(id, &point);
    }

    bool initializeOptimization(int level = 0){
        return optimizer.initializeOptimization(level);
    }

    void optimize(int maxIterations, bool online= false){
        optimizer.optimize(maxIterations, online);
    }

    Eigen::MatrixXd PointGetEstimate(g2o::VertexPointXYZ& point){
        return point.estimate();
    }

    Eigen::MatrixXd PoseGetEstimate(g2o::VertexSE3Expmap& pose){
        g2o::SE3Quat SE3quat = pose.estimate();
        Eigen::Matrix<double,4,4> eigMat = SE3quat.to_homogeneous_matrix();
        return eigMat;

    }

};

void declareBATypes(py::module &m){

    py::class_<g2o::VertexSE3Expmap>(m, "VertexSE3Expmap")
            .def(py::init<>())
            .def("setId", &g2o::VertexSE3Expmap::setId)
            .def("setFixed", &g2o::VertexSE3Expmap::setFixed);

    py::class_<g2o::VertexPointXYZ>(m, "g2o::VertexPointXYZ")
            .def(py::init<>())
            .def("setId", &g2o::VertexPointXYZ::setId)
            .def("setFixed", &g2o::VertexPointXYZ::setFixed)
            .def("setMarginalized", &g2o::VertexPointXYZ::setMarginalized);

    py::class_<g2o::EdgeSE3ProjectXYZ>(m, "g2o::EdgeSE3ProjectXYZ")
            .def(py::init<>())
            .def("setId", &g2o::EdgeSE3ProjectXYZ::setId)
            .def("setVertex", &g2o::EdgeSE3ProjectXYZ::setVertex, "id"_a, "vertex"_a);

    py::class_<BASolver::EdgeReturn>(m, "BASolver_EdgeReturn")
            .def(py::init<>())
            .def_readonly("edge", &BASolver::EdgeReturn::edge)
            .def_readonly("stat", &BASolver::EdgeReturn::stat);

    py::class_<BASolver>(m, "BASolver")
            .def(py::init<>())
            .def("setVerbose", &BASolver::setVerbose)
            .def("addPose", &BASolver::addPose)
            .def("addPoint", &BASolver::addPoint)
            .def("addEdge", py::overload_cast<int, int, float, float, float, float>(&BASolver::addEdge))
            .def("addEdge", py::overload_cast<g2o::VertexPointXYZ&, g2o::VertexSE3Expmap&, float, float, float, float>(&BASolver::addEdge))
            .def("addEdge", py::overload_cast<g2o::EdgeSE3ProjectXYZ&>(&BASolver::addEdge))
            .def("PointSetEstimate", &BASolver::PointSetEstimate)
            .def("PoseSetEstimate", &BASolver::PoseSetEstimate)
            .def("EdgeSetInformation", &BASolver::EdgeSetInformation)
            .def("EdgeSetMeasurement", &BASolver::EdgeSetMeasurement)
            .def("EdgeSetVertex", py::overload_cast<g2o::EdgeSE3ProjectXYZ&, int, g2o::VertexSE3Expmap&>(&BASolver::EdgeSetVertex))
            .def("EdgeSetVertex", py::overload_cast<g2o::EdgeSE3ProjectXYZ&, int, g2o::VertexPointXYZ&>(&BASolver::EdgeSetVertex))
            .def("optimize", &BASolver::optimize, "maxIterations"_a, "online"_a= false)
            .def("initializeOptimization", &BASolver::initializeOptimization, "level"_a=0);

}

#endif //SLAM_PY_TYPES_BA_H
