//
// Created by quan on 2022/7/20.
//

#include <pybind11/pybind11.h>

#include "types_eigen.h"
#include "types_optimizer.h"

#include "types_test.h"
#include "types_dbow.h"

namespace py = pybind11;

PYBIND11_MODULE(slam_py, m) {
    m.attr("package_name") = "slam_py_package";

    declareEigenTypes(m);
    declareOptimizerTypes(m);

    declareTestTypes(m);
    declareDBOWTypes(m);
}