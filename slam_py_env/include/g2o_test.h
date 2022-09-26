//
// Created by quan on 2022/7/20.
//

#ifndef SLAM_PY_TYPES_TEST_H
#define SLAM_PY_TYPES_TEST_H

#include <pybind11/pybind11.h>
#include "pybind11/eigen.h"
#include <iostream>

#include <Eigen/Core>
#include "g2o/core/auto_differentiation.h"
#include "g2o/core/base_unary_edge.h"
#include "g2o/core/base_vertex.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/solvers/eigen//linear_solver_eigen.h"
#include "g2o/core/optimization_algorithm_factory.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/stuff/command_args.h"
#include "g2o/stuff/sampler.h"

using namespace std;
namespace py = pybind11;
using namespace pybind11::literals;

G2O_USE_OPTIMIZATION_LIBRARY(eigen);
G2O_USE_OPTIMIZATION_LIBRARY(dense);

class Test {
public:
    Test() {}

    int add(int x, int y);
};

class VertexParams : public g2o::BaseVertex<3, Eigen::Vector3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    VertexParams() {}

    bool read(std::istream &);

    bool write(std::ostream &) const;

    void setToOriginImpl() {};

    void oplusImpl(const double *update);

};

class EdgePointOnCurve : public g2o::BaseUnaryEdge<1, Eigen::Vector2d, VertexParams> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgePointOnCurve() {};

    virtual bool read(std::istream & /*is*/);

    virtual bool write(std::ostream & /*os*/) const;

    template<typename T>
    bool operator()(const T *params, T *error) const;

    G2O_MAKE_AUTO_AD_FUNCTIONS
};

class DataFitingOlver{
public:
    g2o::SparseOptimizer optimizer;
    g2o::OptimizationAlgorithmProperty solverProperty;

    DataFitingOlver(){};

    void setVerbose(bool verbos){
        this->optimizer.setVerbose(verbos);
    };

    void setAlgorithm(std::string& tag){
        this->optimizer.setAlgorithm(g2o::OptimizationAlgorithmFactory::instance()->construct(tag,solverProperty));
    }

    bool addVertex(VertexParams& vertex){
        return this->optimizer.addVertex(&vertex);
    }

    // todo  python调用无法成功add edge，原因不明
    bool addEdge(EdgePointOnCurve& edge){
        return this->optimizer.addEdge(&edge);
    }

    struct EdgeReturn{
    public:
        EdgePointOnCurve* edge;
        bool stat;
    };
    EdgeReturn* addEdge(VertexParams& vertex, int id){
        EdgePointOnCurve* edge = new EdgePointOnCurve();
        edge->setVertex(id, &vertex);
        bool stat = this->optimizer.addEdge(edge);

        EdgeReturn* res = new EdgeReturn();
        res->edge = edge;
        res->stat = stat;
        return res;
    }

    bool initializeOptimization(int level = 0){
        return optimizer.initializeOptimization(level);
    }

    void optimize(int maxIterations, bool online= false){
        optimizer.optimize(maxIterations, online);
    }

    VertexParams* getVertex(){
        VertexParams* vertex = new VertexParams();
        return vertex;
    }

    EdgePointOnCurve* getEdge(){
        EdgePointOnCurve* edge = new EdgePointOnCurve();
        return edge;
    }

    template<typename T>
    void VertexSetEstimate(VertexParams& vertex, T& data){
        vertex.setEstimate(data);
    }

    template<typename T>
    void EdgeSetInformation(EdgePointOnCurve& edge, T& info){
        edge.setInformation(info);
    }

    template<typename T>
    void EdgeSetMeasurement(EdgePointOnCurve& edge, T& meas){
        edge.setMeasurement(meas);
    }

    Eigen::MatrixXd VertexGetEstimate(VertexParams& vertex){
        return vertex.estimate();
    }

    void EdgeSetVertex(EdgePointOnCurve& edge, int id, VertexParams& vertex){
        edge.setVertex(id, &vertex);
    }

};

void declareTestTypes(py::module &m);

#endif //SLAM_PY_TYPES_TEST_H
