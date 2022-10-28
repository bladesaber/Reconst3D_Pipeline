//
// Created by quan on 2022/10/27.
//

#ifndef DBOW_PYTHON_H
#define DBOW_PYTHON_H

#include "iostream"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "DBoW3.h"

namespace py = pybind11;
using namespace pybind11::literals;

void dbow_print(DBoW3::BowVector &v) {
    std::cout << &v << std::endl;
}

void dbow_print(DBoW3::Vocabulary& voc) {
    std::cout << "[DEBUG] DBOW Vocabulary: Ptr: " << voc << std::endl;
    std::cout << "[DEBUG] DBOW Vocabulary: Words size: " << voc.size() << std::endl;
    std::cout << "[DEBUG] DBOW Vocabulary: Descritor size: " << voc.getDescritorSize() << std::endl;
}

void dbow_print(DBoW3::Database& db) {
    std::cout << "[DEBUG] DBOW Database: Ptr: " << db << std::endl;
    std::cout << "[DEBUG] DBOW Database: DataBase size: " << db.size() << std::endl;
    std::cout << "[DEBUG] DBOW Database: Vocabulary size: " << *db.getVocabulary() << std::endl;
}

void debug(py::array_t<uint8_t>& img){
    py::buffer_info buf = img.request();
    cv::Mat mat(buf.shape[0], buf.shape[1], CV_8UC3, (unsigned char*)buf.ptr);
    cv::imshow("debug", mat);
    cv::waitKey(0);
}

class DBOW3_Library {
public:
    DBOW3_Library() {}

    DBoW3::Vocabulary createVoc(
            int branch_factor, int tree_level,
            DBoW3::WeightingType weight_type, DBoW3::ScoringType score_type,
            bool log= false
            ) {
        DBoW3::Vocabulary voc(branch_factor, tree_level, weight_type, score_type);
        if (log) {
            std::cout << "[Debug] DBOW: branch_factor: " << branch_factor << std::endl;
            std::cout << "[Debug] DBOW: tree_level: " << tree_level << std::endl;
            std::cout << "[Debug] DBOW: WeightingType: " << weight_type << std::endl;
            std::cout << "[Debug] DBOW: ScoringType: " << score_type << std::endl;
            std::cout << "[Debug] DBOW: Create Vocabulary Success" << std::endl;
        }
        return voc;
    };

    DBoW3::Database createDb(DBoW3::Vocabulary& voc, bool use_di = false, int di_level = 0, bool log= false) {
        DBoW3::Database db(voc, use_di, di_level);
        if (log) {
            std::cout << "[Debug] DBOW: Create Database Success" << std::endl;
        }
        return db;
    }

    DBoW3::Database createDb(bool use_di = false, int di_level = 0, bool log= false) {
        DBoW3::Database db(use_di, di_level);
        if (log) {
            std::cout << "[Debug] DBOW: Create Database Success" << std::endl;
        }
        return db;
    }

    void set_Voc2DB(DBoW3::Vocabulary& voc, DBoW3::Database& db) {
        db.setVocabulary(voc);
    }

    DBoW3::Vocabulary loadVoc(std::string &filename, bool log= false) {
        DBoW3::Vocabulary voc;
        voc.load(filename);
        if (log) {
            std::cout << "[Debug] DBOW: Load Vob: " << filename << std::endl;
            std::cout << "[Debug] DBOW: Create Vocabulary Success" << std::endl;
        }
        return voc;
    }

    DBoW3::Database loadDb(std::string &filename, bool log= false) {
        DBoW3::Database db(filename);
        if (log) {
            std::cout << "[Debug] DBOW: Load Database: " << filename << std::endl;
            std::cout << "[Debug] DBOW: Load Database Success" << std::endl;
        }
        return db;
    };

    DBoW3::Database loadDb(DBoW3::Database& db, std::string &filename, bool log= false) {
        db.load(filename);
        if (log) {
            std::cout << "[Debug] DBOW: Load Database: " << filename << std::endl;
            std::cout << "[Debug] DBOW: Load Database Success" << std::endl;
        }
        return db;
    };

    // -----------------------------------------------
    void addVoc(DBoW3::Vocabulary& voc, std::vector<py::array_t<uint8_t>> &features) {
        std::vector<cv::Mat> feature_mat;
        for (int i = 0; i < features.size(); ++i) {
            py::buffer_info buf = features[i].request();
            cv::Mat mat(buf.shape[0], buf.shape[1], CV_8U, (unsigned char*)buf.ptr);
            feature_mat.push_back(mat);
        }
        voc.create(feature_mat);
    }

    void saveVoc(DBoW3::Vocabulary& voc, std::string &filename, bool binary_compressed = true) const {
        voc.save(filename, binary_compressed);
    }

    void clearVoc(DBoW3::Vocabulary& voc) const{
        voc.clear();
    }

    // ------------------------------------------------
    unsigned int addDB(DBoW3::Database& db, py::array_t<uint8_t> &features) {
        py::buffer_info buf = features.request();
        cv::Mat mat(buf.shape[0], buf.shape[1], CV_8U, (unsigned char*)buf.ptr);
        unsigned int idx = db.add(mat);
        return idx;
    }

    unsigned int addDB(DBoW3::Database& db, DBoW3::BowVector& v) {
        unsigned int idx = db.add(v);
        return idx;
    }

    void saveDB(DBoW3::Database& db, std::string &filename) const {
        db.save(filename);
    }

    void clearDB(DBoW3::Database& db) const{
        db.clear();
    }

    // -----------------------------------------------
    double score(DBoW3::Vocabulary& voc, DBoW3::BowVector &v0, DBoW3::BowVector &v1) {
        double score = voc.score(v0, v1);
        return score;
    }

    std::vector<DBoW3::Result> query(DBoW3::Database& db, py::array_t<uint8_t> &features, int max_results) {
        py::buffer_info buf = features.request();
        cv::Mat mat(buf.shape[0], buf.shape[1], CV_8U, (unsigned char*)buf.ptr);

        DBoW3::QueryResults ret;
        db.query(mat, ret, max_results);
        return ret;
    }

    std::vector<DBoW3::Result> query(DBoW3::Database& db, DBoW3::BowVector &v, int max_results) {
        DBoW3::QueryResults ret;
        db.query(v, ret, max_results);
        return ret;
    }

    DBoW3::BowVector transform(DBoW3::Vocabulary& voc, py::array_t<uint8_t>& features){
        py::buffer_info buf = features.request();
        cv::Mat mat(buf.shape[0], buf.shape[1], CV_8U, (unsigned char*)buf.ptr);

        DBoW3::BowVector v;
        voc.transform(mat, v);
        return v;
    }

    DBoW3::BowVector transform(DBoW3::Database& db, py::array_t<uint8_t>& features){
        py::buffer_info buf = features.request();
        cv::Mat mat(buf.shape[0], buf.shape[1], CV_8U, (unsigned char*)buf.ptr);

        DBoW3::BowVector v;
        db.getVocabulary()->transform(mat, v);
        return v;
    }

};

void declareDBOWTypes(py::module &m) {
    m.def("debug", &debug);

    m.def("dbow_print", py::overload_cast<DBoW3::BowVector &>(&dbow_print));
    m.def("dbow_print", py::overload_cast<DBoW3::Vocabulary &>(&dbow_print));
    m.def("dbow_print", py::overload_cast<DBoW3::Database &>(&dbow_print));

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

    py::class_<DBoW3::Result>(m, "DBow_Result")
            .def(py::init<>())
            .def_readonly("Id", &DBoW3::Result::Id)
            .def_readonly("Score", &DBoW3::Result::Score)
            .def_readonly("nWords", &DBoW3::Result::nWords);

    py::class_<DBoW3::Vocabulary>(m, "DBOW3_Vocabulary")
            .def(py::init<>());

    py::class_<DBoW3::Database>(m, "DBOW3_Database")
            .def(py::init<>());

    py::class_<DBOW3_Library>(m, "DBOW3_Library")
            .def(py::init<>())
            .def("createVoc", &DBOW3_Library::createVoc,"branch_factor"_a, "tree_level"_a, "weight_type"_a, "score_type"_a, "log"_a= false)
            .def("loadVoc", &DBOW3_Library::loadVoc, "filename"_a, "log"_a= false)
            .def("createDb", py::overload_cast<DBoW3::Vocabulary&, bool, int, bool>(&DBOW3_Library::createDb), "voc"_a, "use_di"_a, "di_level"_a, "log"_a= false)
            .def("createDb", py::overload_cast<bool, int, bool>(&DBOW3_Library::createDb), "use_di"_a, "di_level"_a, "log"_a= false)
            .def("set_Voc2DB", &DBOW3_Library::set_Voc2DB, "voc"_a, "db"_a)
            .def("loadDb", py::overload_cast<std::string&, bool>(&DBOW3_Library::loadDb), "filename"_a, "log"_a= false)
            .def("loadDb", py::overload_cast<DBoW3::Database&, std::string&, bool>(&DBOW3_Library::loadDb), "db"_a, "filename"_a, "log"_a= false)
            .def("addVoc", &DBOW3_Library::addVoc, "voc"_a, "features"_a)
            .def("saveVoc", &DBOW3_Library::saveVoc, "voc"_a, "filename"_a, "binary_compressed"_a=true)
            .def("clearVoc", &DBOW3_Library::clearVoc, "voc"_a)
            .def("addDB", py::overload_cast<DBoW3::Database&, py::array_t<uint8_t>&>(&DBOW3_Library::addDB), "db"_a, "features"_a)
            .def("addDB", py::overload_cast<DBoW3::Database&, DBoW3::BowVector&>(&DBOW3_Library::addDB), "db"_a, "v"_a)
            .def("saveDB", &DBOW3_Library::saveDB, "db"_a, "filename"_a)
            .def("clearDB", &DBOW3_Library::clearDB, "db"_a)
            .def("score", &DBOW3_Library::score, "voc"_a, "v0"_a, "v1"_a)
            .def("query", py::overload_cast<DBoW3::Database&, py::array_t<uint8_t>&, int>(&DBOW3_Library::query), "db"_a, "features"_a, "max_results"_a)
            .def("query", py::overload_cast<DBoW3::Database&, DBoW3::BowVector&, int>(&DBOW3_Library::query), "db"_a, "v"_a, "max_results"_a)
            .def("transform", py::overload_cast<DBoW3::Vocabulary&, py::array_t<uint8_t>&>(&DBOW3_Library::transform), "voc"_a, "features"_a)
            .def("transform", py::overload_cast<DBoW3::Database&, py::array_t<uint8_t>&>(&DBOW3_Library::transform), "db"_a, "features"_a);

}

#endif //DBOW_PYTHON_H
