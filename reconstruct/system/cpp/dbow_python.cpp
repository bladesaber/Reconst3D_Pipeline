//
// Created by quan on 2022/7/20.
//

#include <pybind11/pybind11.h>
#include "dbow_python.h"

namespace py = pybind11;

PYBIND11_MODULE(dbow_python, m) {
    m.attr("package_name") = "dbow_python";
    declareDBOWTypes(m);
}