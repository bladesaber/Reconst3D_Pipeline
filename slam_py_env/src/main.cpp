//
// Created by quan on 2022/7/20.
//

#include "iostream"
#include "g2o/core/sparse_optimizer.h"
#include "types_test.h"

using namespace std;

void addVertex_test(g2o::SparseOptimizer* opt, VertexParams* vertex){
    opt->addVertex(vertex);
    cout<<"asdad"<<endl;
}

int main(){

//    Test a = Test();
//    int c = a.add(10, 15);
//    cout<<c<<endl;

    VertexParams* params = new VertexParams();
    params->setId(0);
    params->setEstimate(Eigen::Vector3d(1, 1, 1));

    g2o::SparseOptimizer* opt = new g2o::SparseOptimizer();
//    opt->addVertex(params);
    addVertex_test(opt, params);

    cout<<"Finish"<<endl;

    return 0;
}
