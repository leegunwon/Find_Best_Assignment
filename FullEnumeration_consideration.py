import pandas as pd
import itertools
import data_generator

params = {
    "num_job":10,
    "oper_mean": 10,
    "oper_sigma": 2,
    "due_mean": 20,
    "due_sigma": 5,
    "weight_mean": 5,
    "weight_sigma": 1,
    "consideration": ["flow_time"]
}


class DataCalculator():
    def __init__(self, df):
        self.df = df
        self.flow_time = []
        self.makespan = []
        self.tardiness = []
        self.weighted_tardiness = []
        self.time = self.df.loc["소요시간"]
        self.due = self.df.loc["제출기한"]
        self.weight = self.df.loc["성적반영비율"]

    def flow_time_calaulator(self, seq):

        flow_time = [0 for i in range(len(seq))]
        makespan= 0

        for j in seq:
            flow_time[j] = makespan + self.time[j]
            makespan += self.time[j]

        self.flow_time.append(sum(flow_time))
        self.makespan.append(makespan)

    def tardiness_calculator(self, seq):

        tardiness = [0 for i in range(len(seq))]
        makespan = 0

        for j in seq:
            tardiness[j] = max(0, ((makespan + self.time[j]) - self.due[j]))
            makespan += self.time[j]

        self.tardiness.append(sum(tardiness))
        self.makespan.append(makespan)

    def weight_tardiness_calaulator(self, seq):
        makespan = 0
        w_tardiness = 0



        for j in seq:
            w_tardiness += self.weight[j] * max(0, ((makespan + self.time[j]) - self.due[j]))
            makespan += self.time[j]

        self.weighted_tardiness.append(w_tardiness)
        self.makespan.append(makespan)


class FullEnumeration(DataCalculator):
    def __init__(self, df):
        self.df = df
        self.ft_res = []
        self.td_res = []
        self.wt_res = []
        self.flow_time = []
        self.makespan = []
        self.tardiness = []
        self.weighted_tardiness = []
        self.chart = None
        self.df_res = None
        self.time = self.df.loc["소요시간"]
        self.due = self.df.loc["제출기한"]
        self.weight = self.df.loc["성적반영비율"]

    def search(self):
        # 모든 경우의 수
        self.chart = list(itertools.permutations(range(len(self.df.columns)), len(self.df.columns)))
        if "flow_time" in params["consideration"]:

            for i in self.chart:
                self.flow_time_calaulator(i)

            self.ft_min = min(self.flow_time)
            for i in range(len(self.chart)):
                if self.flow_time[i] == self.ft_min:
                    self.ft_res.append(self.chart[i])

        if "tardiness" in params["consideration"]:

            for i in self.chart:
                self.tardiness_calculator(i)

            self.td_min = min(self.tardiness)
            for i in range(len(self.chart)):
                if self.tardiness[i] == self.td_min:
                    self.td_res.append(self.chart[i])

        if "weighted_tardiness" in params["consideration"]:

            for i in self.chart:
                self.weight_tardiness_calaulator(i)

            self.wt_min = min(self.weighted_tardiness)
            for i in range(len(self.chart)):
                if self.weighted_tardiness[i] == self.wt_min:
                    self.wt_res.append(self.chart[i])



    def print_result(self):
        print()
        print("*******************")
        if "flow_time" in params["consideration"]:
            print("min flow_time: ", self.ft_min)
            print("seq flow_time: ", self.ft_res[0])

        if "tardiness" in params["consideration"]:
            print("min tardiness: ", self.td_min)
            print("seq min tardiness: ", self.td_res[0])

        if "makespan" in params["consideration"]:
            print("min makespan: ", self.ms_min)

        if "weighted_tardiness" in params["consideration"]:
            print("min weighted_tardiness: ", self.wt_min)
            print("seq weighted_tardiness: ", self.wt_res[0])
        print("*******************")

def data_process():
    df = pd.read_csv("examdata.csv", encoding="CP949")

    df.set_index(df.iloc[:, 0], inplace=True)
    df = df.iloc[:, 1:]

    return df


def main():
    data_generator.gen_main(params["num_job"], params["oper_mean"], params["oper_sigma"],
                            params["due_mean"], params["due_sigma"],
                            params["weight_mean"],params["weight_sigma"])
    fe = FullEnumeration(data_process())
    fe.search()
    fe.print_result()


if __name__ == "__main__":
    main()
