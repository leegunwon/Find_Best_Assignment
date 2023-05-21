import numpy as np
import pandas as pd

class DataGenerator:
    def __init__(self, lambda_param, num_job,):
        self.lambda_param = lambda_param
        self.num_job = num_job


    def generate(self):
        data = np.random.exponential(scale=1/self.lambda_param, size=self.num_job)

        return data

def main():
    num_job = 5
    oper_time = DataGenerator(5, num_job)
    due_time = DataGenerator(20, num_job)
    arr_time = np.zeros(num_job)
    weight = DataGenerator(10, num_job)

    rand_data = pd.DataFrame((arr_time, oper_time.generate(), due_time.generate(), weight.generate()),
                             index=["출제시간", "소요시간", "제출기한", "성적반영비율"],
                             columns=["J{}".format(i) for i in range(num_job)])
    rand_data.to_csv("examdata.csv", index=True, header=True, encoding="CP949")

if __name__ == "__main__":
    main()