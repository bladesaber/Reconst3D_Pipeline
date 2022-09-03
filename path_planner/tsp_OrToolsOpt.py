import numpy as np
from tqdm import tqdm

from ortools.linear_solver import pywraplp
from ortools.init import pywrapinit

from ortools.sat.python import cp_model

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

class OrtoolsTspOpt(object):

    def create_data(self, dist_graph, start_idx):
        data = {}
        data['distance_matrix'] = dist_graph
        data['num_vehicles'] = 1
        data['depot'] = int(start_idx)

        # data['starts'] = [int(start_idx)]
        # data['ends'] = [int(start_idx)]

        return data

    def single_opt(self, data:dict):
        manager = pywrapcp.RoutingIndexManager(
            ### num citys
            len(data['distance_matrix']),
            ### num cars
            data['num_vehicles'],
            ### start_idx
            data['depot']
        )

        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['distance_matrix'][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )

        def get_print_solution(manager, routing, solution, vehicle_id):
            # print('Objective: {} miles'.format(solution.ObjectiveValue()))
            index = routing.Start(vehicle_id)
            # plan_output = 'Route for vehicle %d:\n'%vehicle_id
            route_distance = 0

            route = [manager.IndexToNode(index)]
            while not routing.IsEnd(index):
                # plan_output += ' {} ->'.format(manager.IndexToNode(index))
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route.append(manager.IndexToNode(index))
                route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)

            # plan_output += ' {}\n'.format(manager.IndexToNode(index))
            # print(plan_output)
            # plan_output += 'Route distance: {}miles\n'.format(route_distance)

            return route, route_distance

        solution = routing.SolveWithParameters(search_parameters)

        if solution:
            route, loss = get_print_solution(manager, routing, solution, vehicle_id=0)
            return True, route, loss
        else:
            return False, None, None

    def run(self, max_iters, dist_graph):
        idxs = np.arange(0, dist_graph.shape[0], 1)
        start_idxs = np.random.choice(idxs, size=min(max_iters, idxs.shape[0]), replace=False)

        best_loss = np.inf
        best_route = None
        for start_id in tqdm(start_idxs):
            data = self.create_data(dist_graph, start_idx=start_id)
            status, route, loss = self.single_opt(data)

            if status:
                print('[DEBUG]: Start From %d loss:%.2f'%(start_id, loss))
                if loss<best_loss:
                    best_loss = loss
                    best_route = route

        return best_route, best_loss



