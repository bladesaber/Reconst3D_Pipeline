//
// Created by quan on 2022/7/20.
//

#include "iostream"
#include "types_test.h"

using namespace std;

int main(){

//    Test a = Test();
//    int c = a.add(10, 15);
//    cout<<c<<endl;

    VertexParams* params = new VertexParams();
    params->setId(0);
    params->setEstimate(Eigen::Vector3d(1, 1, 1));

    Eigen::Vector2d* point = new Eigen::Vector2d(10.0, 20.0);
    EdgePointOnCurve* e = new EdgePointOnCurve;
//    e->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
//    e->setVertex(0, params);
//    e->setMeasurement(point);

    cout<<"Finish"<<endl;

    return 0;
}
