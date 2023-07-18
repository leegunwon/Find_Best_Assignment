import pandas as pd
import random

def data_process():
    df = pd.read_csv("examdata.csv", encoding="CP949")

    df.set_index(df.iloc[:, 0], inplace=True)
    df = df.iloc[:, 1:]

    return df


# 어떻게 짤지
# 1 번째 state는 fitness로 정의
# 2 번째 action은 1~100까지의 정수
# 3 번째 action을 한번 시행하면 시행된 action을 action_list에서 제거
# 4 번째 reward는 fitness의 변화량으로 정의
# 5 번째 done은 action_list가 비었을 때로 정의



class SingleMachine():

    def __init__(self, df):
        self.df = df
        self.action_list = [i for i in range(self.df.shape[1])]
        self.action_num = len(self.action_list)  # 에이전트가 몇 번 움직였는지 기록하는 변수
        self.history = []
        self.action_history = []

    def step(self, a, s):

        fitness = self.cal_tardiness()

        reward = (s - fitness)*self.action_num

        self.action_num -= 1
        done = self.is_done()

        return fitness, reward, done

    def cal_tardiness(self):
        tardiness = [0 for i in range(len(self.action_history))]
        makespan = 0
        due = self.df.loc["제출기한"]
        dur = self.df.loc["소요시간"]
        j = 0
        for i in self.action_history:
            tardiness[j] = max(0, (makespan + dur[i] -due[i]))
            j += 1
            makespan += dur[i]

        return sum(tardiness)

    def reset(self):
        self.action_list = [i for i in range(self.df.shape[1])]
        self.action_num = len(self.action_list)
        self.history = []
        self.action_history = []
        self.fitness = 0
        self.pre_fitness = 0

        return 0

    def is_done(self):
        if (self.action_num == 0):
            return True
        else:
            return False


class QAgent():
    def __init__(self,):
        self.q_table = {}
        self.eps = 0.9
        a = 0

    def select_action(self, s, action_list, action_num):
        # eps-greedy로 액션을 선택해준다
        # 같은 fitness 상태에서 사용했던 action을 회피할 알고리즘 = action_list for문 돌려서 수동으로 최대값 산출
        # q 테이블에 state가 없으면 추가해준다
        if s not in self.q_table.keys():
            self.q_table[s] = {i:0 for i in range(100)}

        coin = random.random()
        if coin < self.eps:
            # q_table에 없는 액션을 취했을 경우 새로운 딕셔너리 추가
            # 랜덤으로 액션을 선택해준다
            action = action_list.pop(random.randint(0, action_num-1))

        else:
            count = 0
            action = 0
            ccount = 0
            maxx = self.q_table[s][action_list[0]]
            for i in action_list:
                if (self.q_table[s][i] >= maxx):
                    maxx = self.q_table[s][i]
                    action = i
                    ccount = count
                count += 1
            action_list.pop(ccount)

        return action

    def update_table(self, history):
        # 한 에피소드에 해당하는 history를 입력으로 받아 q 테이블의 값을 업데이트 한다
        cum_reward = 0
        for transition in history[::-1]:
            s, a, r = transition

            # 몬테 카를로 방식을 이용하여 업데이트.
            cum_reward = cum_reward + r
            self.q_table[s][a] = self.q_table[s][a] + 1 * (cum_reward - self.q_table[s][a])


    def anneal_eps(self):
        self.eps -= 0.0000005
        self.eps = max(self.eps, 0.1)

    def show_table(self):
        key = list(self.q_table.keys())
        for data in key:
            print(f"{data} : 액션{self.q_table[data]}")

def main():
    env = SingleMachine(data_process())
    agent = QAgent()

    for n_epi in range(50000):
        done = False
        s = env.reset()

        while not done:

            a = agent.select_action(s, env.action_list, env.action_num)
            env.action_history.append(a)
            s_prime, r, done = env.step(a, s)
            env.history.append((s, a, r))
            s = s_prime

        agent.update_table(env.history)
        agent.anneal_eps()


    agent.eps = 0

    done = False
    s = env.reset()
    while not done:
        a = agent.select_action(s, env.action_list, env.action_num)
        env.action_history.append(a)
        s_prime, r, done = env.step(a, s)
        env.history.append((s, a, r))
        s = s_prime
    print(env.action_history)
    print(env.cal_tardiness())


if __name__ == '__main__':
    main()
