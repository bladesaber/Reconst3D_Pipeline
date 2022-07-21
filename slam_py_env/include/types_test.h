//
// Created by quan on 2022/7/20.
//

#ifndef SLAM_PY_TYPES_TEST_H
#define SLAM_PY_TYPES_TEST_H

#include <pybind11/pybind11.h>

#include <Eigen/Core>
#include <iostream>

#include "g2o/core/auto_differentiation.h"
#include "g2o/core/base_unary_edge.h"
#include "g2o/core/base_vertex.h"
#include "g2o/core/optimization_algorithm_factory.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/stuff/command_args.h"
#include "g2o/stuff/sampler.h"

using namespace std;
namespace py = pybind11;
using namespace pybind11::literals;

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

void declareTestTypes(py::module &m);

#endif //SLAM_PY_TYPES_TEST_H
