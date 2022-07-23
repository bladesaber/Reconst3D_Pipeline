//
// Created by quan on 2022/7/23.
//

#ifndef SLAM_PY_TYPES_G2O_H
#define SLAM_PY_TYPES_G2O_H

#include "types_test.h"

template<typename T1, typename T2>
void VertexSetEstimate(T1& vertex, T2& data){
    vertex.setEstimate(data);
}

template<typename T1, typename T2>
void EdgeSetInformation(T1& edge, T2& info){
    edge.setInformation(info);
}

template<typename T1, typename T2>
void EdgeSetMeasurement(T1& edge, T2& meas){
    edge.setMeasurement(meas);
}

template<typename T>
Eigen::MatrixXd VertexGetEstimate(T& vertex){
    return vertex.estimate();
}

void declareApplicationTypes(py::module &m) {

    m.def("VertexSetEstimate", &VertexSetEstimate<VertexParams, Eigen::Vector3d>);

    m.def("EdgeSetInformation", &EdgeSetInformation<EdgePointOnCurve, Eigen::Matrix<double, 1, 1>>);

    m.def("EdgeSetMeasurement", &EdgeSetMeasurement<EdgePointOnCurve, Eigen::Vector2d>);

    m.def("VertexGetEstimate", &VertexGetEstimate<VertexParams>);
}

#endif //SLAM_PY_TYPES_G2O_H
