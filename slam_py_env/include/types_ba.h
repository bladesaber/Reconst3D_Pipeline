//
// Created by quan on 2022/7/26.
//

#ifndef SLAM_PY_TYPES_BA_H
#define SLAM_PY_TYPES_BA_H

#include "iostream"
#include "vector"

#include <pybind11/pybind11.h>
#include<pybind11/numpy.h>

#include "Eigen/Core"
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
    EdgeReturn* addEdge(int PointId, int PoseId){
        g2o::EdgeSE3ProjectXYZ* edge = new g2o::EdgeSE3ProjectXYZ();
        edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(PointId)));
        edge->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(PoseId)));

        bool stat = this->optimizer.addEdge(edge);
        EdgeReturn* res = new EdgeReturn();
        res->edge = edge;
        res->stat = stat;
        return res;
    }

    EdgeReturn* addEdge(g2o::VertexPointXYZ& point, g2o::VertexSE3Expmap& pose){
        g2o::EdgeSE3ProjectXYZ* edge = new g2o::EdgeSE3ProjectXYZ();
        edge->setVertex(0, &point);
        edge->setVertex(1, &pose);

        bool stat = this->optimizer.addEdge(edge);
        EdgeReturn* res = new EdgeReturn();
        res->edge = edge;
        res->stat = stat;
        return res;
    }

    void PointSetEstimate(g2o::VertexPointXYZ& point, Eigen::Vector3d& data){
        point.setEstimate(data);
    }

    void PoseSetEstimate(g2o::VertexSE3Expmap& pose, g2o::SE3Quat& data){
        pose.setEstimate(data);
    }

    void EdgeSetInformation(g2o::EdgeSE3ProjectXYZ& edge, Eigen::Matrix2d& info){
        edge.setInformation(info);
    }

    void EdgeSetMeasurement(g2o::EdgeSE3ProjectXYZ& edge, Eigen::Vector2d& meas){
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

    Eigen::Vector3d PointGetEstimate(g2o::VertexPointXYZ& point){
        return point.estimate();
    }

    Eigen::Matrix<double,4,4> PoseGetEstimate(g2o::VertexSE3Expmap& pose){
        g2o::SE3Quat SE3quat = pose.estimate();
        Eigen::Matrix<double,4,4> eigMat = SE3quat.to_homogeneous_matrix();
        return eigMat;

    }

    g2o::SE3Quat ConvertToSE3(Eigen::Matrix<double, 4, 4>& Tcw){
        // Eigen::Matrix<double,3,3> R;
        // R << Tcw(0,0), Tcw(0,1), Tcw(0,2),
        //      Tcw(1,0), Tcw(1,1), Tcw(1,2),
        //      Tcw(2,0), Tcw(2,1), Tcw(2,2);
        // Eigen::Matrix<double,3,1> t(Tcw(0,3), Tcw(1,3), Tcw(2,3));

        // todo ??? error
        // Eigen::Matrix<double,3,3> R = Tcw(Eigen::seq(0, 3), Eigen::seq(0, 3));
        // Eigen::Matrix<double,3,1> t = Tcw(Eigen::all, Eigen::seq(2, 3));

        Eigen::Matrix<double,3,3> R = Tcw.block<3, 3>(0, 0);
        Eigen::Matrix<double,3,1> t = Tcw.block<3, 1>(0, 3);

        g2o::SE3Quat se3 = g2o::SE3Quat(R,t);
        return se3;
    }

};

class PoseOptimizerSolver{
public:
    g2o::SparseOptimizer optimizer;
    g2o::OptimizationAlgorithmLevenberg* solver;

    PoseOptimizerSolver(){
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

    struct EdgeReturn{
    public:
        g2o::EdgeSE3ProjectXYZOnlyPose* edge;
        bool stat;
    };
    EdgeReturn* addEdge(g2o::VertexSE3Expmap& pose, Eigen::Vector3d& point){
        g2o::EdgeSE3ProjectXYZOnlyPose* edge = new g2o::EdgeSE3ProjectXYZOnlyPose();
        edge->setVertex(0, &pose);

        bool stat = this->optimizer.addEdge(edge);
        edge->Xw = point;
        EdgeReturn* res = new EdgeReturn();
        res->edge = edge;
        res->stat = stat;
        return res;
    }

    void PoseSetEstimate(g2o::VertexSE3Expmap& pose, g2o::SE3Quat& data){
        pose.setEstimate(data);
    }

    void EdgeSetInformation(g2o::EdgeSE3ProjectXYZOnlyPose& edge, Eigen::Matrix2d& info){
        edge.setInformation(info);
    }

    void EdgeSetMeasurement(g2o::EdgeSE3ProjectXYZOnlyPose& edge, Eigen::Vector2d& meas){
        edge.setMeasurement(meas);
    }

    bool initializeOptimization(int level = 0){
        return optimizer.initializeOptimization(level);
    }

    void optimize(int maxIterations, bool online= false){
        optimizer.optimize(maxIterations, online);
    }

    Eigen::Matrix<double,4,4> PoseGetEstimate(g2o::VertexSE3Expmap& pose){
        g2o::SE3Quat SE3quat = pose.estimate();
        Eigen::Matrix<double,4,4> eigMat = SE3quat.to_homogeneous_matrix();
        return eigMat;
    }

    g2o::SE3Quat ConvertToSE3(Eigen::Matrix<double, 4, 4>& Tcw){
        // Eigen::Matrix<double,3,3> R;
        // R << Tcw(0,0), Tcw(0,1), Tcw(0,2),
        //      Tcw(1,0), Tcw(1,1), Tcw(1,2),
        //      Tcw(2,0), Tcw(2,1), Tcw(2,2);
        // Eigen::Matrix<double,3,1> t(Tcw(0,3), Tcw(1,3), Tcw(2,3));

        // todo ??? error
        // Eigen::Matrix<double,3,3> R = Tcw(Eigen::seq(0, 3), Eigen::seq(0, 3));
        // Eigen::Matrix<double,3,1> t = Tcw(Eigen::all, Eigen::seq(2, 3));

        Eigen::Matrix<double,3,3> R = Tcw.block<3, 3>(0, 0);
        Eigen::Matrix<double,3,1> t = Tcw.block<3, 1>(0, 3);

        g2o::SE3Quat se3 = g2o::SE3Quat(R,t);
        return se3;
    }

};

void declareBATypes(py::module &m){

    py::class_<g2o::SE3Quat>(m, "SE3Quat")
            .def(py::init<>());

    py::class_<g2o::VertexSE3Expmap>(m, "VertexSE3Expmap")
            .def(py::init<>())
            .def("setId", &g2o::VertexSE3Expmap::setId)
            .def("setFixed", &g2o::VertexSE3Expmap::setFixed);

    py::class_<g2o::VertexPointXYZ>(m, "VertexPointXYZ")
            .def(py::init<>())
            .def("setId", &g2o::VertexPointXYZ::setId)
            .def("setFixed", &g2o::VertexPointXYZ::setFixed)
            .def("setMarginalized", &g2o::VertexPointXYZ::setMarginalized);

    py::class_<g2o::EdgeSE3ProjectXYZ>(m, "EdgeSE3ProjectXYZ")
            .def(py::init<>())
            .def("setId", &g2o::EdgeSE3ProjectXYZ::setId)
            .def("setVertex", &g2o::EdgeSE3ProjectXYZ::setVertex, "id"_a, "vertex"_a)
            .def_readwrite("fx", &g2o::EdgeSE3ProjectXYZ::fx)
            .def_readwrite("fy", &g2o::EdgeSE3ProjectXYZ::fy)
            .def_readwrite("cx", &g2o::EdgeSE3ProjectXYZ::cx)
            .def_readwrite("cy", &g2o::EdgeSE3ProjectXYZ::cy);

    py::class_<BASolver::EdgeReturn>(m, "BASolver_EdgeReturn")
            .def(py::init<>())
            .def_readonly("edge", &BASolver::EdgeReturn::edge)
            .def_readonly("stat", &BASolver::EdgeReturn::stat);

    py::class_<BASolver>(m, "BASolver")
            .def(py::init<>())
            .def("setVerbose", &BASolver::setVerbose)
            .def("addPose", &BASolver::addPose)
            .def("addPoint", &BASolver::addPoint)
            .def("addEdge", py::overload_cast<int, int>(&BASolver::addEdge))
            .def("addEdge", py::overload_cast<g2o::VertexPointXYZ&, g2o::VertexSE3Expmap&>(&BASolver::addEdge), "point"_a, "pose"_a)
            .def("addEdge", py::overload_cast<g2o::EdgeSE3ProjectXYZ&>(&BASolver::addEdge))
            .def("PointSetEstimate", &BASolver::PointSetEstimate)
            .def("PoseSetEstimate", &BASolver::PoseSetEstimate)
            .def("EdgeSetInformation", &BASolver::EdgeSetInformation)
            .def("EdgeSetMeasurement", &BASolver::EdgeSetMeasurement)
            .def("EdgeSetVertex", py::overload_cast<g2o::EdgeSE3ProjectXYZ&, int, g2o::VertexSE3Expmap&>(&BASolver::EdgeSetVertex))
            .def("EdgeSetVertex", py::overload_cast<g2o::EdgeSE3ProjectXYZ&, int, g2o::VertexPointXYZ&>(&BASolver::EdgeSetVertex))
            .def("optimize", &BASolver::optimize, "maxIterations"_a, "online"_a= false)
            .def("initializeOptimization", &BASolver::initializeOptimization, "level"_a=0)
            .def("ConvertToSE3", &BASolver::ConvertToSE3)
            .def("PoseGetEstimate", &BASolver::PoseGetEstimate)
            .def("PointGetEstimate", &BASolver::PointGetEstimate);

    py::class_<PoseOptimizerSolver::EdgeReturn>(m, "PoseOPtimizerSolver_EdgeReturn")
            .def(py::init<>())
            .def_readonly("edge", &PoseOptimizerSolver::EdgeReturn::edge)
            .def_readonly("stat", &PoseOptimizerSolver::EdgeReturn::stat);

    py::class_<g2o::EdgeSE3ProjectXYZOnlyPose>(m, "EdgeSE3ProjectXYZOnlyPose")
            .def(py::init<>())
            .def("setId", &g2o::EdgeSE3ProjectXYZOnlyPose::setId)
            .def_readwrite("fx", &g2o::EdgeSE3ProjectXYZOnlyPose::fx)
            .def_readwrite("fy", &g2o::EdgeSE3ProjectXYZOnlyPose::fy)
            .def_readwrite("cx", &g2o::EdgeSE3ProjectXYZOnlyPose::cx)
            .def_readwrite("cy", &g2o::EdgeSE3ProjectXYZOnlyPose::cy);

    py::class_<PoseOptimizerSolver>(m, "PoseOPtimizerSolver")
            .def(py::init<>())
            .def("setVerbose", &PoseOptimizerSolver::setVerbose)
            .def("addPose", &PoseOptimizerSolver::addPose)
            .def("addEdge", &PoseOptimizerSolver::addEdge, "pose"_a, "point"_a)
            .def("PoseSetEstimate", &PoseOptimizerSolver::PoseSetEstimate)
            .def("EdgeSetInformation", &PoseOptimizerSolver::EdgeSetInformation)
            .def("EdgeSetMeasurement", &PoseOptimizerSolver::EdgeSetMeasurement)
            .def("optimize", &PoseOptimizerSolver::optimize, "maxIterations"_a, "online"_a= false)
            .def("initializeOptimization", &PoseOptimizerSolver::initializeOptimization, "level"_a=0)
            .def("PoseGetEstimate", &PoseOptimizerSolver::PoseGetEstimate)
            .def("ConvertToSE3", &PoseOptimizerSolver::ConvertToSE3);

}

#endif //SLAM_PY_TYPES_BA_H
