cmake_minimum_required(VERSION 3.0.2)
project(slam_py)

#set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_STANDARD 11)
#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")

set(PACKAGE_TOP_DIR ${CMAKE_CURRENT_SOURCE_DIR})
message("PACKAGE_TOP_DIR: " ${PACKAGE_TOP_DIR})

LIST( APPEND CMAKE_MODULE_PATH ${PACKAGE_TOP_DIR}/Thirdparty/g2o/cmake_modules )
set(G2O_ROOT /usr/local/include/g2o)
find_package(G2O REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(OpenCV  REQUIRED)

MESSAGE("OpenCV library status:")
MESSAGE("    version: ${OpenCV_VERSION}")
MESSAGE("    libraries: ${OpenCV_LIBS}")
MESSAGE("    include path: ${OpenCV_INCLUDE_DIRS}")

add_subdirectory(Thirdparty/pybind11)
include_directories(
        Thirdparty/pybind11
        Thirdparty/pybind11/tools
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/core
        ${OpenCV_INCLUDE_DIRS}
        Thirdparty/DBow3/src
)

link_directories(
        ${CERES_INCLUDE_DIRS}
        ${OpenCV_LIBS}
)

pybind11_add_module(slam_py src/slam_py.cpp
        src/types_test.cpp

        Thirdparty/DBow3/src/Vocabulary.cpp
        Thirdparty/DBow3/src/BowVector.cpp
        Thirdparty/DBow3/src/ScoringObject.cpp
        Thirdparty/DBow3/src/Database.cpp
        Thirdparty/DBow3/src/DescManip.cpp
        Thirdparty/DBow3/src/FeatureVector.cpp
        Thirdparty/DBow3/src/QueryResults.cpp
        Thirdparty/DBow3/src/quicklz.c

        )
#add_executable(slam_py src/main.cpp
#        src/types_test.cpp)

target_link_libraries(slam_py PRIVATE
        pybind11::module
        Eigen3::Eigen

        g2o_core
        g2o_stuff
#        g2o_solver_cholmod
        g2o_solver_csparse
        g2o_solver_eigen
        g2o_solver_dense
        g2o_solver_pcg
        g2o_solver_slam2d_linear
        g2o_solver_structure_only
        g2o_types_data
        g2o_types_icp
        g2o_types_sba
        g2o_types_sclam2d
        g2o_types_sim3
        g2o_types_slam2d
        g2o_types_slam2d_addons
        g2o_types_slam3d
        g2o_types_slam3d_addons

        ${CERES_LIBRARIES}
        ${OpenCV_LIBS}
        )

