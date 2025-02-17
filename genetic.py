# 一様交差で評価関数はフィットネス、フォールド確率の調整を世代交代ごとに設定(h１の時0.5~0.7、、h2の時0.3~0.5)
import random
from pypokerengine.api.game import setup_config, start_poker
from heuristicAI import HeuristicPlayer
import helper
import numpy as np

init_def_prob = np.array([
    [0.6, 0.2, 0.0, 0.2],
    [0.4, 0.4, 0.1, 0.1],
    [0.1, 0.7, 0.2, 0.0],
    [0.0, 0.6, 0.4, 0.0],
    [0.0, 0.3, 0.7, 0.0],
])

def normalize(narray):
    return [x / sum(x) for x in narray]

class Population(object):
    def __init__(self, size):
        self.pop = []  # AIプレイヤーを格納するリスト
        self.size = size  # AIプレイヤーの数
        for i in range(size):
            # Generate a random bot
            def_prob = normalize(init_def_prob * (1 + np.random.uniform(-0.25, 0.25, size=(5, 4))))
            self.pop.append(HeuristicPlayer(def_prob, agg=np.random.uniform(0, 2)))

    def compute_weighted_fitness(self):
        """
        各プレイヤーのfitnessを評価関数とし、新しい選択確率を計算。
        """
        base_fitness = self.compute_fitness()  # 現在のfitness（掛け金）を取得
        total = sum(base_fitness)
        return [f / total for f in base_fitness]  # 正規化して選択確率として返す

    def crossover(self, parent1, parent2):
        """
        2つの親個体から新しい子個体を生成する交差操作
        """
        child_prob = []  # 子個体の行動確率行列
        for i in range(5):  # 各行をランダムに親から選択
            if random.uniform(0, 1) > 0.5:
                child_prob.append(parent1.default_prob[i])
            else:
                child_prob.append(parent2.default_prob[i])

        # 攻撃性をランダムに親から継承し、少しノイズを加える
        child_aggression = (parent1.aggression + parent2.aggression) / 2 + np.random.uniform(-0.1, 0.1)
        return HeuristicPlayer(np.array(child_prob), agg=child_aggression)

    def adjust_fold_probabilities(self):
        """
        フォールド確率の調整:
        - 苦戦している手 (H1): 0.5 ~ 0.7
        - 悪い手 (H2): 0.3 ~ 0.5
        """
        for player in self.pop:
            # 苦戦している手 (H1): 行動確率を設定
            fold_prob_h1 = random.uniform(0.5, 0.7)
            remaining_h1 = 1.0 - fold_prob_h1
            other_probs_h1 = np.random.dirichlet(np.ones(3)) * remaining_h1
            player.default_prob[0] = [fold_prob_h1, *other_probs_h1]

            # 悪い手 (H2): 行動確率を設定
            fold_prob_h2 = random.uniform(0.3, 0.5)
            remaining_h2 = 1.0 - fold_prob_h2
            other_probs_h2 = np.random.dirichlet(np.ones(3)) * remaining_h2
            player.default_prob[1] = [fold_prob_h2, *other_probs_h2]

    def birth_cycle(self):
        """
        フィットネスのみを考慮した世代交代
        """
        weighted_fitnesses = self.compute_weighted_fitness()
        print("Weighted fitnesses")
        print(weighted_fitnesses)

        # 親個体（生存者）の選択
        new_generation = list(np.random.choice(self.pop, self.size // 2, p=weighted_fitnesses, replace=False))

        # 交差による新しい子個体の生成
        for _ in range(self.size - len(new_generation)):  # 残りの個体を生成
            parent1, parent2 = np.random.choice(new_generation, 2, replace=False)  # ランダムに2つの親個体を選択
            child = self.crossover(parent1, parent2)  # 子個体を生成
            if random.uniform(0, 1) > 0.75:  # 突然変異を適用
                child.mutate()
            new_generation.append(child)

        self.pop = new_generation  # 次世代の集団を更新
        self.adjust_fold_probabilities()  # 苦戦している手と悪い手の確率を調整
        print("New generation created!")

    def compute_fitness(self):
        """
        各プレイヤーのフィットネス（掛け金の結果）を計算
        """
        total_fitness = [0] * self.size

        for rnd in range(5):  # 5ラウンドのゲームを実行
            print(f"Beginning population round {rnd}")
            tables = np.random.permutation(self.size)  # ランダムにプレイヤーをシャッフル

            # プレイヤーを4つのテーブルに分ける
            table1 = [(self.pop[i], i) for i in tables[:self.size // 4]]
            table2 = [(self.pop[i], i) for i in tables[self.size // 4:2 * self.size // 4]]
            table3 = [(self.pop[i], i) for i in tables[2 * self.size // 4:3 * self.size // 4]]
            table4 = [(self.pop[i], i) for i in tables[3 * self.size // 4:]]

            # 各テーブルでのゲーム結果からフィットネスを計算
            round_fitness = helper.add([
                self.play_round(table1),
                self.play_round(table2),
                self.play_round(table3),
                self.play_round(table4)
            ])
            print(f"The fitness totals for round {rnd} are: {round_fitness}")

            # 各ラウンドのフィットネスの結果を累積
            total_fitness = helper.add([round_fitness, total_fitness])
            print(total_fitness)

        return total_fitness  # 全てのラウンドの結果を返す

    def play_round(self, players):
        """
        各ラウンドをプレイし、結果を収集
        """
        config = setup_config(max_round=20, initial_stack=200, small_blind_amount=1)
        print("Setting up a new table")
        for player, num in players:
            print(f"Welcoming player {num}")
            config.register_player(name=num, algorithm=player)

        results = start_poker(config, verbose=0)
        print("The final results of the poker tournament are: ", results)

        fitnesses = [0] * self.size
        for player in results['players']:
            fitnesses[player['name']] = player['stack']  # プレイヤーのスタックを保存
        return fitnesses

    def print(self):
        print([(x.default_prob, x.aggression) for x in self.pop])


# 実行部分
a = Population(20)
a.print()
for epoch in range(5):
    print(f"Running epoch {epoch}")
    a.birth_cycle()
    a.print()
