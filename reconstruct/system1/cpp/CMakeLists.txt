cmake_minimum_required(VERSION 3.0.2)
project(dbow_python)

#set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_STANDARD 11)
#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")

set(PACKAGE_TOP_DIR ${CMAKE_CURRENT_SOURCE_DIR})
message("PACKAGE_TOP_DIR: " ${PACKAGE_TOP_DIR})

find_package(Eigen3 REQUIRED)
find_package(OpenCV  REQUIRED)

MESSAGE("OpenCV library status:")
MESSAGE("    version: ${OpenCV_VERSION}")
MESSAGE("    libraries: ${OpenCV_LIBS}")
MESSAGE("    include path: ${OpenCV_INCLUDE_DIRS}")

set(CPP_Thirdparty_DIR /home/quan/Desktop/company/Reconst3D_Pipeline/slam_py_env/Thirdparty)
set(SO_OUT_DIR /home/quan/Desktop/company/Reconst3D_Pipeline/reconstruct/system1/cpp/lib)
add_subdirectory(${CPP_Thirdparty_DIR}/pybind11 ${SO_OUT_DIR})
include_directories(
        ${CPP_Thirdparty_DIR}/pybind11
        ${CPP_Thirdparty_DIR}/pybind11/tools
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${OpenCV_INCLUDE_DIRS}
        ${CPP_Thirdparty_DIR}/DBow3/src
)

link_directories(
        ${OpenCV_LIBS}
        lib
)

pybind11_add_module(dbow_python dbow_python.cpp
        ${CPP_Thirdparty_DIR}/DBow3/src/Vocabulary.cpp
        ${CPP_Thirdparty_DIR}/DBow3/src/BowVector.cpp
        ${CPP_Thirdparty_DIR}/DBow3/src/ScoringObject.cpp
        ${CPP_Thirdparty_DIR}/DBow3/src/Database.cpp
        ${CPP_Thirdparty_DIR}/DBow3/src/DescManip.cpp
        ${CPP_Thirdparty_DIR}/DBow3/src/FeatureVector.cpp
        ${CPP_Thirdparty_DIR}/DBow3/src/QueryResults.cpp
        ${CPP_Thirdparty_DIR}/DBow3/src/quicklz.c
        )
#add_executable(slam_py src/main.cpp
#        src/types_test.cpp)

target_link_libraries(dbow_python PRIVATE
        pybind11::module
        Eigen3::Eigen
        ${OpenCV_LIBS}
        )

