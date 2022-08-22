import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sko.GA import GeneticAlgorithmBase
from sko.tools import set_run_mode

class GA_TSP(GeneticAlgorithmBase):
    def __init__(
            self,
            func, n_dim,
            size_pop=50, max_iter=200, prob_mut=0.001
    ):
        super().__init__(func, n_dim, size_pop=size_pop, max_iter=max_iter, prob_mut=prob_mut)
        self.has_constraint = False
        self.len_chrom = self.n_dim

        self.crtbp()

    def crtbp(self):
        self.Chrom = np.random.rand(self.size_pop, self.len_chrom)
        self.Chrom = self.Chrom.argsort(axis=1) + 1
        return self.Chrom

    def ranking(self):
        self.FitV = -self.Y

    def selection(self, tourn_size=3):
        aspirants_idx = np.random.randint(self.size_pop, size=(self.size_pop, tourn_size))
        aspirants_values = self.FitV[aspirants_idx]
        winner = aspirants_values.argmax(axis=1)  # winner index in every team
        sel_index = [aspirants_idx[i, j] for i, j in enumerate(winner)]
        self.Chrom = self.Chrom[sel_index, :]
        return self.Chrom

    def crossover(self):
        Chrom, size_pop, len_chrom = self.Chrom, self.size_pop, self.len_chrom
        for i in range(0, size_pop, 2):
            Chrom1, Chrom2 = self.Chrom[i], self.Chrom[i + 1]
            cxpoint1, cxpoint2 = np.random.randint(0, self.len_chrom - 1, 2)
            if cxpoint1 >= cxpoint2:
                cxpoint1, cxpoint2 = cxpoint2, cxpoint1 + 1
            # crossover at the point cxpoint1 to cxpoint2
            pos1_recorder = {value: idx for idx, value in enumerate(Chrom1)}
            pos2_recorder = {value: idx for idx, value in enumerate(Chrom2)}
            for j in range(cxpoint1, cxpoint2):
                value1, value2 = Chrom1[j], Chrom2[j]
                pos1, pos2 = pos1_recorder[value2], pos2_recorder[value1]
                Chrom1[j], Chrom1[pos1] = Chrom1[pos1], Chrom1[j]
                Chrom2[j], Chrom2[pos2] = Chrom2[pos2], Chrom2[j]
                pos1_recorder[value1], pos1_recorder[value2] = pos1, j
                pos2_recorder[value1], pos2_recorder[value2] = j, pos2

            self.Chrom[i], self.Chrom[i + 1] = Chrom1, Chrom2
        return self.Chrom

    def reverse(self, individual):
        n1, n2 = np.random.randint(0, individual.shape[0] - 1, 2)
        if n1 >= n2:
            n1, n2 = n2, n1 + 1
        individual[n1:n2] = individual[n1:n2][::-1]
        return individual

    def mutation(self):
        for i in range(self.size_pop):
            if np.random.rand() < self.prob_mut:
                self.Chrom[i] = self.reverse(self.Chrom[i])
        return self.Chrom

    def chrom2x(self, Chrom):
        return Chrom

    def x2y(self):
        self.Y_raw = self.func(self.X)
        self.Y = self.Y_raw
        return self.Y

    def run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        for i in tqdm(range(self.max_iter)):
            Chrom_old = self.Chrom.copy()
            self.X = self.Chrom
            self.Y = self.x2y()
            self.ranking()
            self.selection()
            self.crossover()
            self.mutation()

            # put parent and offspring together and select the best size_pop number of population
            self.Chrom = np.concatenate([Chrom_old, self.Chrom], axis=0)
            self.X = self.Chrom
            self.Y = self.x2y()
            self.ranking()
            selected_idx = np.argsort(self.Y)[:self.size_pop]
            self.Chrom = self.Chrom[selected_idx, :]

            # record the best ones
            generation_best_index = self.FitV.argmax()
            self.generation_best_X.append(self.X[generation_best_index, :].copy())
            self.generation_best_Y.append(self.Y[generation_best_index])
            self.all_history_Y.append(self.Y.copy())
            self.all_history_FitV.append(self.FitV.copy())

        global_best_index = np.array(self.generation_best_Y).argmin()
        self.best_x = self.generation_best_X[global_best_index]
        self.best_y = self.func(np.array([self.best_x]))
        return self.best_x, self.best_y

num_points = 49
points_coordinate = np.random.rand(num_points, 2)
points_coordinate = np.concatenate((np.array([[0, 0]]), points_coordinate), axis=0)
start_pos = np.array([0])

def cal_total_distance(routine):
    idx_array = np.concatenate((start_pos, routine[:-1]))
    from_idx = idx_array[:-1]
    to_idx = idx_array[1:]

    from_pos = points_coordinate[from_idx]
    to_pos = points_coordinate[to_idx]

    return np.mean(np.sqrt(np.sum(np.power(from_pos - to_pos, 2), axis=1)))

set_run_mode(cal_total_distance, 'multiprocessing')
model = GA_TSP(func=cal_total_distance, n_dim=49, size_pop=3000, max_iter=1000)
best_routine, best_distance = model.run()

best_routine = np.concatenate((start_pos, best_routine))
fig, ax = plt.subplots(1, 2)
best_points_coordinate = points_coordinate[best_routine, :]
ax[0].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
ax[1].plot(model.generation_best_Y)
plt.show()
