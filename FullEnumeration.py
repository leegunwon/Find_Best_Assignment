import pandas as pd
import itertools
import data_generator

params = {
    "num_job":6,
    "oper_mean": 10,
    "oper_sigma": 2,
    "due_mean": 20,
    "due_sigma": 5,
    "weight_mean": 5,
    "weight_sigma": 1,
    "consideration": ["flow_time", "tardiness", "weighted_tardiness"]
}


class DataCalculator():
    def __init__(self, df):
        self.df = df
        self.flow_time = []
        self.makespan = []
        self.tardiness = []
        self.weighted_tardiness = []

    def calaulator(self, seq):

        flow_time = [0 for i in range(len(seq))]
        tardiness = [0 for i in range(len(seq))]
        makespan= 0
        w_tardiness = 0

        time = self.df.loc["소요시간"]
        due = self.df.loc["제출기한"]
        weight = self.df.loc["성적반영비율"]


        for j in seq:
            flow_time[j] = makespan + time[j]
            tardiness[j] = max(0, ((makespan + time[j]) - due[j]))
            makespan += time[j]
            w_tardiness += weight[j] * tardiness[j]



        self.flow_time.append(sum(flow_time))
        self.tardiness.append(sum(tardiness))
        self.makespan.append(makespan)
        self.weighted_tardiness.append(w_tardiness)

    def find_min(self):

        self.ft_min = min(self.flow_time)
        self.td_min = min(self.tardiness)
        self.ms_min = min(self.makespan)
        self.wt_min = min(self.weighted_tardiness)



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

    def search(self):
        # 모든 경우의 수
        self.chart = list(itertools.permutations(range(len(self.df.columns)), len(self.df.columns)))
        # 모든 경우의 수로 데이터 프레임 만들기 (데이터는
        for i in self.chart:
            self.calaulator(i)

        self.find_min()

        for i in range(len(self.chart)):
            if self.flow_time[i] == self.ft_min:
                self.ft_res.append(self.chart[i])
            if self.tardiness[i] == self.td_min:
                self.td_res.append(self.chart[i])
            if self.weighted_tardiness[i] == self.wt_min:
                self.wt_res.append(self.chart[i])


    def print_result(self):
        print()
        print("*******************")
        if "flow_time" in params["consideration"]:
            print("min flow_time: ", self.ft_min)
            print("seq flow_time: ", self.ft_res)

        if "tardiness" in params["consideration"]:
            print("min tardiness: ", self.td_min)
            print("seq min tardiness: ", self.td_res)

        if "makespan" in params["consideration"]:
            print("min makespan: ", self.ms_min)

        if "weighted_tardiness" in params["consideration"]:
            print("min weighted_tardiness: ", self.wt_min)
            print("seq weighted_tardiness: ", self.wt_res)
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
