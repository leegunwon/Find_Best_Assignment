import pandas as pd
import random

def data_process():
    df = pd.read_csv("examdata.csv", encoding="CP949")

    df.set_index(df.iloc[:, 0], inplace=True)
    df = df.iloc[:, 1:]

    return df


params = {'scenario':5000,
          'consideration':10,
          'eps_decay':0.000001}


class SingleMachine():

    def __init__(self, df):
        self.df = df
        self.due = self.df.loc["제출기한"]
        self.dur = self.df.loc["소요시간"]
        self.makespan = 0
        self.action_list = [[i,self.df.loc["소요시간"][i]] for i in range(self.df.shape[1])]
        self.action_list.sort(key=lambda x: x[1], reverse=False)
        self.action_num = len(self.action_list)  # 에이전트가 몇 번 움직였는지 기록하는 변수
        self.history = []
        self.action_history = []
        self.hist = 0
        self.best_tardiness = 0

    def step(self, a, s):

        fitness = self.cal_tardiness(a)

        reward = self.hist - fitness

        self.hist = fitness

        self.action_num -= 1

        done = self.is_done()

        return fitness, reward, done

    def cal_tardiness(self, a):

        tardiness = self.hist + max(0, (self.makespan + self.dur[a] -self.due[a]))
        self.makespan += self.dur[a]

        return tardiness

    def reset(self):
        self.action_list = [[i,self.df.loc["소요시간"][i]] for i in range(self.df.shape[1])]
        self.action_list.sort(key=lambda x: x[1], reverse=False)
        self.action_num = len(self.action_list)
        self.history = []
        self.action_history = []
        self.hist = 0
        self.makespan = 0


        return 0

    def is_done(self):
        if (self.action_num == 0):
            return True
        else:
            return False


class QAgent():
    def __init__(self):
        self.q_table = {}
        self.eps = 0.9
        a = 0

    def select_action(self, s, action_list, action_num):
        # eps-greedy로 액션을 선택해준다
        # 같은 fitness 상태에서 사용했던 action을 회피할 알고리즘 = action_list for문 돌려서 수동으로 최대값 산출
        # q 테이블에 state가 없으면 추가해준다
        if (s,action_num) not in self.q_table.keys():
            if action_num > params['consideration']:
                self.q_table[s,action_num] = {action_list[i][0]:-100 for i in range(params['consideration'])}

            else:
                self.q_table[s,action_num] = {action_list[i][0]:-100 for i in range(action_num)}

        if action_num > params['consideration']:
            num =params['consideration']-1
        else:
            num = action_num - 1

        coin = random.random()
        if coin < self.eps:
            # q_table에 없는 액션을 취했을 경우 새로운 딕셔너리 추가
            # 랜덤으로 액션을 선택해준다

            action = action_list[random.randint(0, num)]
            action_list.remove(action)

            if action[0] not in self.q_table[s, action_num].keys():
                self.q_table[s, action_num][action[0]] = -100
        else:
            if action_list[0][0] not in self.q_table[s, action_num].keys():
                self.q_table[s, action_num][action_list[0][0]] = -100

            action = action_list[0]

            for j in range(num+1):
                if action_list[j][0] not in self.q_table[s, action_num].keys():
                    self.q_table[s, action_num][action_list[j][0]] = -100
            # q_table에 있는 액션 중 가장 큰 값을 가진 액션을 선택해준다
                if self.q_table[s, action_num][action_list[j][0]] > self.q_table[s, action_num][action[0]]:
                    action = action_list[j]

            action_list.remove(action)

        return action[0]

    def update_table(self, history):
        # 한 에피소드에 해당하는 history를 입력으로 받아 q 테이블의 값을 업데이트 한다
        cum_reward = 0
        action_num = 1
        for transition in history[::-1]:
            s, a, r = transition
            # 몬테 카를로 방식을 이용하여 업데이트.
            cum_reward = cum_reward + r
            self.q_table[s,action_num][a] = self.q_table[s,action_num][a] + 1 * (cum_reward - self.q_table[s,action_num][a])
            action_num += 1

    def anneal_eps(self):
        self.eps -= params['eps_decay']
        self.eps = max(self.eps, 0.1)

    def show_table(self):
        key = list(self.q_table.keys())
        for data in key:
            print(f"{data} : 액션{self.q_table[data]}")

def main():
    env = SingleMachine(data_process())
    agent = QAgent()

    for n_epi in range(params['scenario']):
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
    print(env.hist)


if __name__ == '__main__':
    for i in range(5):
        main()
