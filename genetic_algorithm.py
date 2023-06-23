import FullEnumeration_consideration as FE
import pandas as pd
import numpy as np
import random
import data_generator
import time

params = {
    "num_job": 8,
    "oper_mean": 10,
    "oper_sigma": 2,
    "due_mean": 20,
    "due_sigma": 5,
    "weight_mean": 5,
    "weight_sigma": 1,
    # "tardiness", "weighted_tardiness", "flow_time", "makespan"
    "consideration": ["flow_time"],
    'MUT': 0.5,  # 변이확률(%)
    'END': 0.8,  # 설정한 비율만큼 chromosome이 수렴하면 탐색을 멈추게 하는 파라미터 (%)
    'POP_SIZE': 10,  # population size 10 ~ 100
    'NUM_OFFSPRING': 5,  # 한 세대에 발생하는 자식 chromosome의 수
    'SELECTION_PRESSURE': 3  # 1보다 큰 정수
}


class GA():
    def __init__(self):

        self.df = FE.data_process()
        self.population = []
        self.fitness = []
        self.num_offspring = params["NUM_OFFSPRING"]
        self.pop_size = params["POP_SIZE"]
        self.num_job = params["num_job"]
        self.time = self.df.loc["소요시간"]
        self.due = self.df.loc["제출기한"]
        self.weight = self.df.loc["성적반영비율"]


    def initial_population(self):
        self.population = []

        for i in range(self.pop_size):
            self.population.append(random.sample(range(self.num_job), self.num_job))


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


    def weight_tardiness_calaulator(self, chromosome):
        weight_tardiness = 0
        makespan = 0

        for j in chromosome:
            weight_tardiness += self.weight[j] * max(0, ((makespan + self.time[j]) - self.due[j]))
            makespan += self.time[j]

        return weight_tardiness


    def flow_time_get_fitness(self):
        flow_time_fitness = []

        for chrom in self.population:
            flow_time_fitness.append([chrom, self.flow_time_calaulator(chrom)])

        return flow_time_fitness


    def tardiness_get_fitness(self):
        tardiness_fitness = []

        for i in range(len(self.population)):
            tardiness_fitness.append([self.population[i], self.tardiness_calculator(self.population[i])])

        return tardiness_fitness


    def weighted_tardiness_get_fitness(self):
        weighted_tardiness_fitness = []

        for i in range(len(self.population)):
            weighted_tardiness_fitness.append([self.population[i], self.weight_tardiness_calaulator(self.population[i])])

        return weighted_tardiness_fitness


    def print_average_fitness(self, fitness):
        population_average_fitness = 0

        for i in range(len(fitness)):
            population_average_fitness += fitness[i][1]

        print(population_average_fitness / len(fitness))


    def sort_population(self, fitness):
        fitness.sort(key=lambda x: x[1], reverse=False)
        for i in range(len(fitness)):
            self.population[i] = fitness[i][0]
        return fitness


    def selection_operater(self, fitness):
        """
        품질 비례 룰렛휠
        :param fitness: [job sequence[(int) * num_jobs], fitness(float)]
        self.mom_ch: [job sequence[(int) * num_jobs]]
        self.dad_ch: [job sequence[(int) * num_jobs]]
        """

        selection = [(fitness[-1][1] - fitness[i][1]) + (fitness[-1][1] - fitness[0][1])/(params['SELECTION_PRESSURE']-1)
                     for i in range(len(fitness))]

        temp1 = 0
        temp2 = 0

        while(temp1 == temp2):
            temp1 = np.random.randint(0, sum(selection))
            temp2 = np.random.randint(0, sum(selection))

        while(temp1 > 0):
            j = 0
            temp1 -= selection[j]

        while (temp2 > 0):
            k = 0
            temp2 -= selection[k]

        self.mom_ch = fitness[j][0]
        self.dad_ch = fitness[k][0]


    def crossover_operater(self):
        """
        싸이클 교차
        :return: self.offspring_ch [offsprings[job sequence[(int) * num_job] * num_offsprings]]
        """
        self.offspring_ch = []

        for i in range(self.num_offspring):
            sample = list(range(self.num_job))
            chromosome = [i for i in range(self.num_job)]

            for j in sample:
                gene = self.mom_ch[j]
                chromosome[j] = gene

                while(self.mom_ch[j] != gene):
                    num = sample.pop(self.dad_ch.index(gene))    # 아빠 해에서 선택된 gene과 동일한 값의 gene 있는 위치를 찾는다
                    gene = self.mom_ch[num]
                    chromosome[num] = gene          # 그 위치에 엄마 해의 gene값을 자식 해에 유전한다.

            cromosome = self.mutation_operater(chromosome)   # 랜덤 숫자로 슬라이싱한 chromosome을 mutation_operater에 넣어줌
            self.offspring_ch.append(chromosome)


    def mutation_operater(self, chromosome):
        # 랜덤에 걸릴 경우 chromosome의 앞뒤 자리를 바꿔줌
        for i in range(len(chromosome)):
            num = np.random.randint(10)
            if len(chromosome) == 1:
                pass
            else:
                if num >= (params["MUT"]*10):
                    pass
                else:
                    if i == 0:
                        chromosome[i], chromosome[i + 1] = chromosome[i + 1], chromosome[i]
                    else:
                        chromosome[i], chromosome[i - 1] = chromosome[i - 1], chromosome[i]

        return chromosome


    def replacement_operator(self):

        self.population = self.population[0:(self.pop_size - self.num_offspring)] + self.offspring_ch


    def print_result(self, fitness, generation):

        print(f"탐색이 완료되었습니다. \t 최종 세대수: {generation},\t 최종 해: {fitness[0][0]},\t 최종 적합도: {fitness[0][1]}")


    def search(self):

        generation = 0
        # 초기 population 생성
        self.initial_population()

        if "flow_time" in params["consideration"]:
            while True:
                count = 1

                w_fitness = self.flow_time_get_fitness()
                fitness = self.sort_population(w_fitness)

               # self.print_average_fitness(fitness)

                self.selection_operater(fitness)
                self.crossover_operater()
                self.replacement_operator()

                for i in range(self.pop_size - 1):
                    if fitness[i][1] == fitness[i + 1][1]:
                        count += 1
                if (count == (params["END"] * 10)):
                    self.print_result(fitness, generation)

                    break
                generation += 1

        if "tardiness" in params["consideration"]:
            while True:
                count = 1
                w_fitness = self.tardiness_get_fitness()
                fitness = self.sort_population(w_fitness)

                self.print_average_fitness(fitness)

                self.selection_operater(fitness)
                self.crossover_operater()
                self.replacement_operator()

                for i in range(len(fitness)-1):
                    if fitness[i][1] == fitness[i+1][1]:
                        count += 1

                if (count == params["END"]*10):
                    self.print_result(fitness, generation)

                    break

                generation += 1

        if "weighted_tardiness" in params["consideration"]:
            while True:
                count = 1

                w_fitness = self.weighted_tardiness_get_fitness()
                fitness = self.sort_population(w_fitness)

                # 평균 값 출력
               # self.print_average_fitness(fitness)

                self.selection_operater(fitness)
                self.crossover_operater()
                self.replacement_operator()

                for i in range(len(fitness)-1):
                    if fitness[i][1] == fitness[i+1][1]:
                        count += 1
                if (count == params["END"]*10):
                    self.print_result(fitness, generation)

                    break

                generation += 1


def main():

    # 데이터를 생성하는 코드 (FullEnumeration과 동일한 자료로 코드를 돌리기 위해 주석)
    # data_generator.gen_main(params["num_job"], params["oper_mean"], params["oper_sigma"],
    #                         params["due_mean"], params["due_sigma"],
    #                         params["weight_mean"],params["weight_sigma"])

    ga = GA()
    ga.search()



if __name__ == "__main__":
    t1 =time.time()
    FE.params["num_job"] = params["num_job"]
    FE.main()
    t2 = time.time()
    main()
    t3 = time.time()
    print(f"FullEnumeration: {t2-t1}, GA: {t3-t2}")










