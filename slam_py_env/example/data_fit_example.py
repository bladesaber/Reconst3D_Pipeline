import slam_py_env.build.slam_py as slam_py
import numpy as np
import random
import math

def test_func(x, a, b, lambda_v):
    y = a * math.exp(-lambda_v * x) + b
    return y

def main():
    numPoints = 50
    maxIterations = 10
    verbose = True

    a = 2.
    b = 0.4
    lambda_v = 0.2

    points = []
    for idx in range(numPoints):
        x = random.uniform(0, 10.0)
        y = test_func(x, a, b, lambda_v)
        points.append(np.array([x, y]))

    taskSolver = slam_py.DataFitingOlver()
    taskSolver.setAlgorithm("lm_dense")

    params = taskSolver.getVertex()
    params.setId(0)
    taskSolver.VertexSetEstimate(params, np.array([1.0, 1.0, 1.0]))
    stat = taskSolver.addVertex(params)
    print("[Debug]: Add Vertex %d"%stat)

    for idx, point in enumerate(points):
        edgeRes = taskSolver.addEdge(params, 0)
        if edgeRes.stat:
            edge = edgeRes.edge
            edge.setId(idx)
            taskSolver.EdgeSetInformation(edge, np.array([1.0]).reshape((1, 1)))
            taskSolver.EdgeSetMeasurement(edge, point)

    slam_py.PrintfGraphInfo(taskSolver.optimizer)

    taskSolver.setVerbose(verbose)
    taskSolver.initializeOptimization()
    taskSolver.optimize(maxIterations)

    res = taskSolver.VertexGetEstimate(params)

    print("Target curve")
    print("a * exp(-lambda * x) + b")
    print("Iterative least squares solution")
    print("a = %f"%(res[0]))
    print("b = %f"%(res[1]))
    print("lambda = %f"%(res[2]))

if __name__ == '__main__':
    main()