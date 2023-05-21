import pandas as pd
import numpy as np
import itertools

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
        weight = df.iloc[3]

        return sum(tardiness * weight)

class FullEnumeration(DataCalculator):
    def __init__(self, df):
        self.df = df
        self.data_list = []
        self.column_num = len(self.df.columns)


    def search(self):
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

        res_data = [[self.cal_flow_time(self.data_list[i]), sum(self.cal_tardiness(self.data_list[i]))] for i in
                    range(len(self.data_list))]
        flow_data, tardiness_data = zip(*res_data)

        # flow_data에서 제일 큰 값의 리스트에서 위치
        numbers = [i for i in range(len(res_data)) if res_data[i][0] == min(flow_data)]
        numbers2 = [i for i in range(len(res_data)) if res_data[i][1] == min(tardiness_data)]
        print("flow time이 가장 작은 값의 index")
        for i in numbers:
            print(pd.DataFrame(self.data_list[i]).index)
        print("tardiness가 가장 작은 값의 index")
        for i in numbers2:
            print(pd.DataFrame(self.data_list[i]).index)
        print("둘다 가장 작은 값의 index")
        for i in set(numbers).intersection(set(numbers2)):
            print(pd.DataFrame(self.data_list[i]).index)


        # 모든 경우의 수로 순서 배열하는 알고리즘 짜줘


def data_process():
    df = pd.read_csv("examdata.csv", encoding="CP949")

    df.set_index(df.iloc[:, 0], inplace=True)
    df = df.iloc[:, 1:]

    return df
def main():
    # 첫번째 열 추출

    fe = FullEnumeration(data_process())
    fe.search()



if __name__ == "__main__":
    main()