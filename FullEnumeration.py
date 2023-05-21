import pandas as pd
import numpy as np
import itertools
import data_generator

params = {
    "num_job": 5,
    "oper_lambda_param": 5,
    "due_lambda_param": 10,
    "weight_lambda_param": 5,
    "consideration" : ["flow_time", "tardiness", "weighted_tardiness"]
}
class DataCalculator:

    def cal_flow_time(self, data):
        df = pd.DataFrame(data).T
        time = df.loc["소요시간"]
        flow_time = np.zeros(len(time))
        for i in range(len(time)):
            if i == 0:
                flow_time[i] = time[i]
            else:
                flow_time[i] = flow_time[i - 1] + time[i]

        return sum(flow_time)

    def cal_makespan(self, data):
        df = pd.DataFrame(data).T
        time = df.loc["소요시간"]

        return sum(time)

    def cal_tardiness(self, data):
        df = pd.DataFrame(data).T
        time = df.loc["소요시간"]
        tardiness = np.zeros(len(time))
        due = self.df.loc["제출기한"]

        for i in range(len(time)):
            tardiness[i] = max(0, due[i] - time[i])

        return tardiness

    def total_weighted_tardiness(self, data, tardiness):
        df = pd.DataFrame(data).T
        weight = df.loc["성적반영비율"]

        return sum(tardiness * weight)

class FullEnumeration(DataCalculator):
    def __init__(self, df):
        self.df = df
        self.data_list = []
        self.column_num = len(self.df.columns)


    def search(self, consideration):
        # 모든 경우의 수
        chart = list(itertools.permutations(range(self.column_num), self.column_num))
        # 모든 경우의 수로 데이터 프레임 만들기 (데이터는


        for i in chart:
            e = 0
            self.datas = list(range(self.column_num))
            for j in i:
                self.datas[e] =  self.df.iloc[:, j]
                e += 1
            self.data_list.append(self.datas)

        if "flow_time" in consideration:
            res_data = [self.cal_flow_time(self.data_list[i]) for i in
                        range(len(self.data_list))]

            numbers1 = [i for i in range(len(res_data)) if res_data[i] == min(res_data)]

            # 출력
            print("flow time이 가장 작은 값의 index")
            for i in numbers1:
                print(pd.DataFrame(self.data_list[i]).index)

        if "tardiness" in consideration:
            tardiness_data = [self.cal_tardiness(self.data_list[i]) for i in
                              range(len(self.data_list))]
            sum_tardiness_data = [sum(tardiness_data[i]) for i in range(len(tardiness_data))]
            numbers2 = [i for i in range(len(sum_tardiness_data))
                        if sum_tardiness_data[i] == min(sum_tardiness_data)]

            # 출력
            print("tardiness가 가장 작은 값의 index")
            for i in numbers2:
                print(pd.DataFrame(self.data_list[i]).index)

        if "weighted_tardiness" in consideration:
            if "tardiness" not in consideration:
                tardiness_data = [self.cal_tardiness(self.data_list[i]) for i in
                                  range(len(self.data_list))]

            weighted_tardiness_data = [self.total_weighted_tardiness(self.data_list[i], tardiness_data[i]) for i in
                                       range(len(self.data_list))]

            numbers3 = [i for i in range(len(weighted_tardiness_data))
                        if weighted_tardiness_data[i] == min(weighted_tardiness_data)]
            # 출력
            print("weighted_tardiness가 가장 작은 값의 index")
            for i in numbers3:
                print(pd.DataFrame(self.data_list[i]).index)



        # 모든 경우의 수로 순서 배열하는 알고리즘 짜줘


def data_process():
    df = pd.read_csv("examdata.csv", encoding="CP949")

    df.set_index(df.iloc[:, 0], inplace=True)
    df = df.iloc[:, 1:]

    return df
def main():
    data_generator.gen_main(params["num_job"], params["oper_lambda_param"],
                            params["due_lambda_param"], params["weight_lambda_param"])

    fe = FullEnumeration(data_process())
    fe.search(params["consideration"])



if __name__ == "__main__":
    main()