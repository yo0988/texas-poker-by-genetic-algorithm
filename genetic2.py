## A genetic algorithm for finding optimal heuristic AI parameters
#フォールドの数値を調節している(消極的なGA)
import random
from pypokerengine.api.game import setup_config, start_poker
from heuristicAI import HeuristicPlayer
from consoleAI import ConsolePlayer
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
            # generate a random bot
            # 最小値の乱数-0.25最大値の乱数0.25で5行4列で作成。元の行動確率に対して±25%の範囲で変動
            def_prob = normalize(init_def_prob * (1 + np.random.uniform(-0.25, 0.25, size=(5, 4))))
            # 0から2の範囲でランダムに攻撃率を設定
            self.pop.append(HeuristicPlayer(def_prob, agg=np.random.uniform(0, 2)))

    def birth_cycle(self):
        """ Conduct a full Moran process, storing the relative fitnesses in a file """
        fitnesses = np.sqrt(self.compute_fitness())
        fitnesses = fitnesses / sum(fitnesses)
        print("fitness")
        print(fitnesses)
        print(self.size)

        # フィットネスに基づいて集団の半数(selfsize/2)を選択（生存者）。
        new_generation = list(np.random.choice(self.pop, self.size // 2, p=fitnesses, replace=False))  # モランプロセスの生存者

        # フィットネスに基づいて新たな個体を追加
        births = np.random.choice(self.pop, self.size - self.size // 2, p=fitnesses, replace=True)

        # 新たに選ばれたAIに突然変異を加える
        for new_ai in births:
            if np.random.uniform(0, 1) > 0.75:  # 突然変異の確率は25%
                new_ai.mutate()  # AIパラメーターがランダムに変更
            new_generation.append(new_ai)  # 次世代に追加

        self.apply_high_fold_probability_to_first(new_generation)  # 最初のプレイヤーのフォールド確率を設定
        self.pop = new_generation  # 次世代の集団を更新
        print("New generation created!")

    def apply_high_fold_probability_to_first(self, generation):
        """
        最初のAIプレイヤーが30%の確率でフォールド確率を1.0に設定。
        """
        if random.uniform(0, 1) <= 0.3:  # 30%の確率
            first_player = generation[0]  # 最初のプレイヤーを選択
            for i in range(2):  # 全ての行動確率（5行分）を更新
                first_player.default_prob[i] = [1.0, 0.0, 0.0, 0.0]  # フォールドのみを1.0に設定
            print("The first AI player's fold probability was set to 1.0")

    def compute_fitness(self):
        """
        Divide all the players into 4 tables to play. Play a total of 5 rounds.
        """
        total_fitness = [0] * self.size  # 各プレイヤーの総合フィットネスを0で初期化
        for rnd in range(5):
            print("Beginning population round {0}".format(rnd))  # ランダムにプレイヤーをシャッフル
            tables = np.random.permutation(self.size)  # ランダムにプレイヤーをシャッフル

            # プレイヤーを4つのテーブルに分ける
            table1 = [(self.pop[i], i) for i in tables[:self.size // 4]]
            table2 = [(self.pop[i], i) for i in tables[self.size // 4:2 * self.size // 4]]
            table3 = [(self.pop[i], i) for i in tables[2 * self.size // 4:3 * self.size // 4]]
            table4 = [(self.pop[i], i) for i in tables[3 * self.size // 4:]]

            # 各テーブルでのゲーム結果からフィットネスを計算し、プレイヤーごとのフィットネスを合計
            round_fitness = helper.add([self.play_round(table1), self.play_round(table2), self.play_round(table3), self.play_round(table4)])
            print("The fitness totals for this round are: ", round_fitness)

            # 各ラウンドのフィットネスの結果を蓄積
            total_fitness = helper.add([round_fitness, total_fitness])
            print(total_fitness)

        return total_fitness  # 全てのラウンドの結果をもとにしたフィットネスを返す

    def play_round(self, players):
        """
        Input: players is a list of tuples (player, num) where num is the index of player in self.pop
        Output: a list of their payoffs
        """
        # ポーカーゲームの設定（ゲームラウンドは最大20ラウンド、200の初期スタック、SBは1）
        config = setup_config(max_round=20, initial_stack=200, small_blind_amount=1)
        print("Setting up a new table")
        for player, num in players:
            print("Welcoming player {0}".format(num))
            config.register_player(name=num, algorithm=player)
        results = start_poker(config, verbose=0)  # ポーカーを実施し、その結果を保存
        print("The final results of the poker tournament are: ", results)

        fitnesses = [0] * self.size
        for player in results['players']:
            fitnesses[player['name']] = player['stack']  # 各プレイヤーの名前と最終スタックを取得し保存する
        return fitnesses

    def print(self):
        print([(x.default_prob, x.aggression) for x in self.pop])


a = Population(20)
a.print()
for epoch in range(5):
    print("Running epoch {0}".format(epoch))
    a.birth_cycle()
    a.print()
