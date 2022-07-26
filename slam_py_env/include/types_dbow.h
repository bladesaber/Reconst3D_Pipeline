//
// Created by quan on 2022/7/25.
//

#ifndef SLAM_PY_TYPES_DBOW_H
#define SLAM_PY_TYPES_DBOW_H

#include "iostream"
#include <pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include "vector"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "DBoW3.h"

namespace py = pybind11;
using namespace pybind11::literals;

void float_array_test(py::array_t<float>& data){
    py::buffer_info buf = data.request();
    std::cout<<*data<<std::endl;
}

void opencv_test(py::array_t<uint8_t>& img){
    py::buffer_info buf = img.request();
    cv::Mat mat(buf.shape[0], buf.shape[1], CV_8UC3, (unsigned char*)buf.ptr);
    cv::imshow("debug", mat);
    cv::waitKey(0);
}

class DBOW3_Library{
public:
    DBoW3::Vocabulary voc;
    std::vector<cv::Mat> features;

    DBoW3::Database db;

    DBOW3_Library(){};

    void createVoc(int branch_factor, int tree_level, DBoW3::WeightingType weight_type, DBoW3::ScoringType score_type){
        std::cout<<"[Debug]: branch_factor: "<< branch_factor<<std::endl;
        std::cout<<"[Debug]: tree_level: "<< tree_level<<std::endl;
        std::cout<<"[Debug]: WeightingType: "<< weight_type<<std::endl;
        std::cout<<"[Debug]: ScoringType: "<< score_type<<std::endl;
        voc = DBoW3::Vocabulary(branch_factor, tree_level, weight_type, score_type);
        std::cout<<"[Debug]: Create Vocabulary Success"<<std::endl;

        db = DBoW3::Database(voc, false, 0);
        std::cout<<"[Debug]: Create Database Success"<<std::endl;
    };

    void createVoc(std::string& filename){
        std::cout<<"[Debug]: Load Vob: "<< filename<<std::endl;
        voc.load(filename);
        std::cout<<"[Debug]: Create Vocabulary Success"<<std::endl;
    };

    void createDb(bool use_di=false, int di_level=0){
        db = DBoW3::Database(voc, use_di, di_level);
        std::cout<<"[Debug]: Create Database Success"<<std::endl;
    }

    void createDb(std::string& filename){
        std::cout<<"[Debug]: Load Database: "<< filename<<std::endl;
        db = DBoW3::Database(filename);
        std::cout<<"[Debug]: Load Database Success"<<std::endl;
    };

    void addFeature(py::array_t<uint8_t>& feat){
        py::buffer_info buf = feat.request();
        std::cout<<"[Debug]: Add Feature shape: "<<buf.shape[0]<<"x"<<buf.shape[1]<<std::endl;
        cv::Mat feat_mat(buf.shape[0], buf.shape[1], CV_8U, (unsigned char*)buf.ptr);

//        std::cout<<feat_mat(cv::Range(0, 1), cv::Range::all())<<std::endl;
        features.push_back(feat_mat);
    }

    void clearFeatures(){features.clear();}
    void clearVoc(){voc.clear();}
    void clearDb(){db.clear();}

    void updateVoc(){voc.create(features);}
    void saveVoc(std::string& filename, bool binary= true) const{voc.save(filename, binary);}
    void saveDb(std::string& filename) const{db.save(filename);}

    DBoW3::BowVector& vocTransform(py::array_t<uint8_t>& feat, DBoW3::BowVector& v1) const{
        py::buffer_info buf = feat.request();
        cv::Mat feat_mat(buf.shape[0], buf.shape[1], CV_8U, (unsigned char*)buf.ptr);
        voc.transform(feat_mat, v1);
        return v1;
    }

    double vocScore(DBoW3::BowVector& v1, DBoW3::BowVector& v2) const{return voc.score(v1, v2);}

    unsigned int dbAddFeature(py::array_t<uint8_t>& feat){
        py::buffer_info buf = feat.request();
        cv::Mat feat_mat(buf.shape[0], buf.shape[1], CV_8U, (unsigned char*)buf.ptr);
        return db.add(feat_mat);
    }

    std::vector<DBoW3::Result> dbQuery(py::array_t<uint8_t>& feat, int max_results) const{
        py::buffer_info buf = feat.request();
        cv::Mat feat_mat(buf.shape[0], buf.shape[1], CV_8U, (unsigned char*)buf.ptr);

        DBoW3::QueryResults ret;
        db.query(feat_mat, ret, max_results);
        return ret;
    }

};

void declareDBOWTypes(py::module &m){
    m.def("float_array_test", &float_array_test);
    m.def("opencv_test", &opencv_test);

    py::enum_<DBoW3::WeightingType>(m, "Voc_WeightingType")
            .value("TF_IDF", DBoW3::WeightingType::TF_IDF)
            .value("TF", DBoW3::WeightingType::TF)
            .value("IDF", DBoW3::WeightingType::IDF)
            .value("BINARY", DBoW3::WeightingType::BINARY)
            .export_values();

    py::enum_<DBoW3::ScoringType>(m, "Voc_ScoringType")
            .value("L1_NORM", DBoW3::ScoringType::L1_NORM)
            .value("L2_NORM", DBoW3::ScoringType::L2_NORM)
            .value("CHI_SQUARE", DBoW3::ScoringType::CHI_SQUARE)
            .value("KL", DBoW3::ScoringType::KL)
            .value("BHATTACHARYYA", DBoW3::ScoringType::BHATTACHARYYA)
            .value("DOT_PRODUCT", DBoW3::ScoringType::DOT_PRODUCT)
            .export_values();

    py::class_<DBoW3::BowVector>(m, "BowVector")
            .def(py::init<>());
    py::class_<DBoW3::Result>(m, "Result")
            .def(py::init<>())
            .def_readonly("Id", &DBoW3::Result::Id)
            .def_readonly("Score", &DBoW3::Result::Score)
            .def_readonly("nWords", &DBoW3::Result::nWords);

    py::class_<DBOW3_Library>(m, "DBOW3_Library")
            .def(py::init<>())
            .def("createVoc", py::overload_cast<int, int, DBoW3::WeightingType, DBoW3::ScoringType>(&DBOW3_Library::createVoc))
            .def("createVoc", py::overload_cast<std::string&>(&DBOW3_Library::createVoc))
            .def("createDb", py::overload_cast<bool, int>(&DBOW3_Library::createDb), "use_di"_a=false, "di_level"_a=0)
            .def("createDb", py::overload_cast<std::string&>(&DBOW3_Library::createDb))
            .def("addFeature", &DBOW3_Library::addFeature)
            .def("clearFeatures", &DBOW3_Library::clearFeatures)
            .def("clearVoc", &DBOW3_Library::clearVoc)
            .def("clearDb", &DBOW3_Library::clearDb)
            .def("updateVoc", &DBOW3_Library::updateVoc)
            .def("saveVoc", &DBOW3_Library::saveVoc, "filename"_a, "binary"_a=true)
            .def("saveDb", &DBOW3_Library::saveDb)
            .def("vocTransform", &DBOW3_Library::vocTransform)
            .def("vocScore", &DBOW3_Library::vocScore)
            .def("dbAddFeature", &DBOW3_Library::dbAddFeature)
            .def("dbQuery", &DBOW3_Library::dbQuery);
}

#endif //SLAM_PY_TYPES_DBOW_H
