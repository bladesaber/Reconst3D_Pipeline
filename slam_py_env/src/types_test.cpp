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

