import pandas as pd
import numpy as np
import random
import time
import plotly.graph_objs as go


params = {
    "num_job": 100,
    "oper_mean": 10,
    "oper_sigma": 2,
    "due_mean": 20,
    "due_sigma": 5,
    "weight_mean": 5,
    "weight_sigma": 1,
    # "tardiness", "weighted_tardiness", "flow_time", "makespan"
    "consideration": ["tardiness"],
    'MUT': 0.8,  # 변이확률(%)
    'END': 0.8,  # 설정한 비율만큼 chromosome이 수렴하면 탐색을 멈추게 하는 파라미터 (%)
    'POP_SIZE': 50,  # population size 10 ~ 100
    'NUM_OFFSPRING': 14,  # 한 세대에 발생하는 자식 chromosome의 수
    'SELECTION_PRESSURE': 3  # 1보다 큰 정수 1< x < 10
}

def data_process():
    df = pd.read_csv("examdata.csv", encoding="CP949")

    df.set_index(df.iloc[:, 0], inplace=True)
    df = df.iloc[:, 1:]

    return df

class GA():
    def __init__(self):
        self.sequence = []
        self.n_population = []
        self.population = [[0]*params["num_job"] for i in range(params["POP_SIZE"])]
        self.fitnesses = []
        self.generations = []
        self.counts = []
        self.offspring_ch = []
        self.generation = 0
        self.df = data_process()
        self.fitness = []
        self.mutation = params["MUT"]
        self.num_offspring = params["NUM_OFFSPRING"]
        self.pop_size = params["POP_SIZE"]
        self.num_job = params["num_job"]
        self.time = self.df.loc["소요시간"]
        self.due = self.df.loc["제출기한"]
        self.weight = self.df.loc["성적반영비율"]
        self.selection_pressure = params["SELECTION_PRESSURE"]
        self.minimize = 100000
        self.min_sol = None

    def initial_population(self):
        for i in range(self.pop_size):
            self.n_population.append([[random.random() for j in range(self.num_job)], 0])


    def population_sequence(self):
        n_pop = [0] * self.pop_size

        for k in range(self.pop_size):
            for j in range(self.num_job):
                if self.n_population[k][0].count(self.n_population[k][0][j]) > 1:
                    self.n_population[k][0][j] += 0.00001

        for i in range(self.pop_size):
            n_pop[i] = sorted(self.n_population[i][0])

        for k in range(self.pop_size):
            for j in range(self.num_job):
                seq = self.n_population[k][0].index(n_pop[k][j])

                self.population[k][seq] = j
            self.n_population[k][1] = self.tardiness_calculator(self.population[k])


    def flow_time_calaulator(self, chromosome):
        flow_time = [0 for i in range(len(chromosome))]
        makespan = 0

        for j in chromosome:
            flow_time[j] = makespan + self.time[j]
            makespan += self.time[j]

        return sum(flow_time)

    def tardiness_calculator(self, chromosome):
        tardiness = [0 for i in range(len(chromosome))]
        makespan = 0

        for j in chromosome:
            tardiness[j] = max(0, ((makespan + self.time[j]) - self.due[j]))
            makespan += self.time[j]

        return sum(tardiness)



    def sort_population(self, fitness):
        fitness.sort(key=lambda x: x[1], reverse=False)

        return fitness

    def roulette_wheel_selection(self, fitness):
        """
        품질 비례 룰렛휠
        :param fitness: [job sequence[(int) * num_jobs], fitness(float)]
        self.mom_ch: [job sequence[(int) * num_jobs]]
        self.dad_ch: [job sequence[(int) * num_jobs]]
        """
        selection = [(fitness[-1][1] - fitness[i][1]) + (fitness[-1][1] - fitness[0][1]) / (params['SELECTION_PRESSURE'] - 1)
                    for i in range(len(fitness))]
        while(True):
            if(sum(selection) == 0):
                p = 0
                k = 0
                break
            temp1 = np.random.randint(0, sum(selection))
            temp2 = np.random.randint(0, sum(selection))

            for i in range(len(selection)):
                temp1 -= selection[i]
                if temp1 <= 0:
                    p = i
                    break

            for i in range(len(selection)):
                temp2 -= selection[i]
                if temp2 <= 0:
                    k = i
                    break

            if p != k:
                break

        self.mom_ch = self.n_population[p]
        self.dad_ch = self.n_population[k]


    def part_crossover(self):

        slice = random.randint(0, self.num_job)
        chromosome = self.mom_ch[0][:slice] + self.dad_ch[0][slice:]
        if self.mutation > random.random():
            chromosome = self.scramble_mutation(chromosome)
        self.offspring_ch.append([chromosome,0])


    def sum_crossover(self):

        chromosome = [(self.mom_ch[0][i] + self.dad_ch[0][i])/2 for i in range(self.num_job)]
        if self.mutation > random.random():
            chromosome = self.scramble_mutation(chromosome)
        self.offspring_ch.append([chromosome,0])


    def uniform_crossover(self):

        chromosome = [self.mom_ch[0][i] if random.random() < 0.5 else self.dad_ch[0][i] for i in range(self.num_job)]
        if self.mutation > random.random():
            chromosome = self.scramble_mutation(chromosome)
        self.offspring_ch.append([chromosome,0])


    def multiple_crossover(self):
        chromosome = [self.mom_ch[0][i] * self.dad_ch[0][i] for i in range(self.num_job)]
        if self.mutation > random.random():
            chromosome = self.exchange_mutation(chromosome)
        self.offspring_ch.append([chromosome,0])

    def exchange_mutation(self, chromosome):
        """
        EM
        :param chromosome: [job sequence[(int) * num_job]]
        :return: chromosome: [job sequence[(int) * num_job]]
        """
        slice = random.sample(range(self.num_job), 2)
        chromosome[slice[0]], chromosome[slice[1]] = chromosome[slice[1]], chromosome[slice[0]]

        return chromosome

    def scramble_mutation(self, chromosome):
        """
        SM
        :param chromosome:
        :return:
        """
        slice = random.sample(range(self.num_job), 2)
        slice.sort()
        n_chromosome = chromosome[0:slice[0]] + chromosome[slice[1]:self.num_job]
        num = np.random.randint(0, len(n_chromosome))
        part = chromosome[slice[0]:slice[1]]
        random.shuffle(part)
        for i in part:
            n_chromosome.insert(num, i)

        return n_chromosome


    def replacement(self):
        self.n_population.sort(key=lambda x: x[1], reverse=False)
        self.n_population = self.n_population[0:(self.pop_size - self.num_offspring)] + self.offspring_ch


    def print_result(self, fitness):

        print(f"탐색이 완료되었습니다. \t 최종 세대수: {self.generation},\t 최종 해: {fitness[0][0]},\t 최종 적합도: {fitness[0][1]}")




    def search(self):
        self.initial_population()
        if "tardiness" in params["consideration"]:
            while True:
                count = 0
                self.population_sequence()
                fitness = self.sort_population(self.n_population)

                self.offspring_ch = []      # 자손 초기화
                for i in range(self.num_offspring):  # 자손 생성
                    self.roulette_wheel_selection(fitness)
                    self.uniform_crossover()
                self.replacement()


                for i in range(self.pop_size - 1):
                    if fitness[i][1] == fitness[i+1][1]:
                        count += 1

                if self.generation % 100 == 0:
                    self.counts.append(count)
                    self.generations.append(self.generation)
                    self.fitnesses.append(fitness[0][1])

                if(self.minimize > fitness[0][1]):
                    self.minimize = fitness[0][1]
                    self.min_sol = fitness[0][0]
                    self.minn = self.fitness

                if (count >= (params["END"] * self.pop_size)):
                    self.print_result(fitness)
                    print(f"최소값은{self.minimize, self.min_sol}")
                    print(self.minn)
                    break



                if (time.time() - t1 > 595):
                    self.print_result(fitness)
                    print(f"최소값은{self.minimize, self.min_sol}")
                    break

                self.generation += 1


def main():
    ga = GA()
    ga.search()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ga.generations, y=ga.counts, mode='lines', name="COUNTS"))
    fig.update_layout(scene=dict(xaxis_title='Generation', yaxis_title='Count'), xaxis=dict(tickfont=dict(size=50)), yaxis=dict(tickfont=dict(size=50)))
    fig.show()
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=ga.generations, y=ga.fitnesses, mode='lines', name="FITNESS"))
    fig1.update_layout(scene=dict(xaxis_title='Generation', yaxis_title='Fitness'), xaxis=dict(tickfont=dict(size=50)), yaxis=dict(tickfont=dict(size=50)))
    fig1.show()


if __name__ == "__main__":
    t1 = time.time()
    main()

