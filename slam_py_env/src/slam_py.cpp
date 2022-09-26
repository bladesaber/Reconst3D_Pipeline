//
// Created by quan on 2022/7/20.
//

#include <pybind11/pybind11.h>

#include "types_optimizer.h"
#include "types_dbow.h"
#include "types_ba.h"

namespace py = pybind11;

PYBIND11_MODULE(slam_py, m) {
    m.attr("package_name") = "slam_py_package";

    declareOptimizerTypes(m);

    declareDBOWTypes(m);
    declareBATypes(m);
}