import slam_py_env.build.slam_py as slam_py
import numpy as np
import random
import math

def test_func(x, a, b, lambda_v):
    y = a * math.exp(-lambda_v * x) + b
    return y

def main():
    numPoints = 50
    maxIterations = 20
    verbose = True

    a = 2.
    b = 0.4
    lambda_v = 0.2

    points = []
    for idx in range(numPoints):
        x = random.uniform(0, 10.0)
        y = test_func(x, a, b, lambda_v)
        points.append(np.array([x, y]))

    optimizer = slam_py.SparseOptimizer()
    solverProperty = slam_py.OptimizationAlgorithmProperty()
    slam_py.SetAlgorithm(optimizer, "lm_dense", solverProperty)

    params = slam_py.VertexParams()
    params.setId(0)
    slam_py.VertexSetEstimate(params, np.array([1.0, 1.0, 1.0]))
    stat = optimizer.addVertex(params)
    if not stat:
        raise ValueError("Add Vertex Fail")

    for idx, point in enumerate(points):
        # edge = slam_py.EdgePointOnCurve()
        # edge.setId(idx)
        # slam_py.EdgeSetInformation(edge, np.array([1.0]).reshape((1, 1)))
        # slam_py.EdgeSetVertex(edge, 0, params)
        # slam_py.EdgeSetMeasurement(edge, point)
        # stat = optimizer.addEdge(edge)

        # edge.setVertex(0, params)
        # stat = optimizer.addEdge(edge)

        a = slam_py.OptAddEdge(optimizer, params)
        print(a)

    slam_py.PrintfGraphInfo(optimizer)

    # optimizer.set_verbose(verbose)
    # optimizer.initializeOptimization()
    # optimizer.optimize(maxIterations)
    #
    # res = slam_py.VertexGetEstimate(params)
    #
    # print("Target curve")
    # print("a * exp(-lambda * x) + b")
    # print("Iterative least squares solution")
    # print("a = %f"%(res[0]))
    # print("b = %f"%(res[1]))
    # print("lambda = %f"%(res[2]))

def test():

    # slam_py.getMatrix(10, 10, [1,2,3,4,5])

    # a = slam_py.Test()

    # a = slam_py.Vector3d(15, 50, 30)
    vertex1 = slam_py.VertexParams()
    slam_py.VertexSetEstimate(vertex1, np.array([1.0, 1.0, 1.0]))
    # a = vertex1.estimate()
    a = slam_py.VertexGetEstimate(vertex1)
    print(a)

    # n = slam_py.vertex_test(vertex1, np.array([1,1,1]))

    # slam_py.vertex_test(vertex1, a)

    # vertex1.setEstimate(a)
    # vertex1.setId(0)

    # edge1 = slam_py.EdgePointOnCurve()
    # edge1.setId(0)
    # slam_py.EdgeaddVertex(edge1, vertex1, 0)
    #
    # opt = slam_py.SparseOptimizer()
    # stat = opt.addVertex(vertex1)
    # print(stat)
    # stat = opt.addEdge(edge1)
    # print(stat)
    # opt.initializeOptimization(0)

    # print('???')

if __name__ == '__main__':
    # test()
    main()