import numpy as np
import pandas as pd

class DataGenerator:
    def __init__(self, mean, sigma, num_job,):
        self.mean = mean
        self.num_job = num_job
        self.sigma = sigma


    def generate(self):
        data = np.random.normal(self.mean, self.sigma, size=self.num_job)
        data = data.astype(int)

        return  data

def gen_main(num_job, oper_mean, oper_sigma, due_mean, due_sigma, weight_mean, weight_sigma):
    oper_time = DataGenerator(oper_mean, oper_sigma, num_job)
    due_time = DataGenerator(due_mean, due_sigma, num_job)
    arr_time = np.zeros(num_job)
    weight = DataGenerator(weight_mean, weight_sigma, num_job)

    rand_data = pd.DataFrame((arr_time, oper_time.generate(), due_time.generate(), weight.generate()),
                             index=["출제시간", "소요시간", "제출기한", "성적반영비율"],
                             columns=["J{}".format(i) for i in range(num_job)])
    rand_data.to_csv("examdata.csv", index=True, header=True, encoding="CP949")

if __name__ == "__main__":
    gen_main()