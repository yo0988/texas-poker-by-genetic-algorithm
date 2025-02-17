import random
import numpy as np

class HeuristicPlayer:
    def __init__(self, def_prob, agg=1):
        """
        def_prob: 5x4 matrix of default probabilities for fold, call, raise, bluff.
        agg: Aggression level.
        """
        self.aggression = agg
        self.default_prob = def_prob

# フォールド確率を0.1〜0.6に設定し、プレイヤーの確率設定に反映する関数
def assign_random_fold_probability(player):
    fold_prob = random.uniform(0.1, 0.6)
    player.default_prob[0][0] = fold_prob  # 0行目の最初の要素をフォールド確率と仮定
    return fold_prob

# 5x4のデフォルト確率行列を生成
init_def_prob = np.array([
    [0.1, 0.2, 0.3, 0.4],  # フォールド確率がここに設定される
    [0.4, 0.4, 0.1, 0.1],
    [0.1, 0.7, 0.2, 0.0],
    [0.0, 0.6, 0.4, 0.0],
    [0.0, 0.3, 0.7, 0.0]
])

# プレイヤー（AI）インスタンスを生成
player = HeuristicPlayer(init_def_prob)

# フォールド確率を0.1〜0.6に設定して割り当てる
new_fold_prob = assign_random_fold_probability(player)

# 結果を出力
print(f"Assigned fold probability: {new_fold_prob}")
print("Updated probability matrix (default_prob):")
print(player.default_prob)
