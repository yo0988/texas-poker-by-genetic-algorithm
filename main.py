#各GAの評価実験プログラミング。インポートしたいGAを選択する。この例は評価関数における実験
from pypokerengine.api.game import setup_config, start_poker
from heuristicAI import HeuristicPlayer
from consoleAI import ConsolePlayer
from genetic import Population as PopulationA  # 一様交差で評価関数はフィットネス、フォールドの数値を0.1~0.6に変更
from genetic3 import Population as PopulationB  #一様交差で評価関数は攻撃率✖️フィットネス

# 初期行動確率行列を定義
init_def_prob = [
    [0.6, 0.2, 0.0, 0.2],
    [0.4, 0.4, 0.1, 0.1],
    [0.1, 0.7, 0.2, 0.0],
    [0.0, 0.6, 0.4, 0.0],
    [0.0, 0.3, 0.7, 0.0]
]

# 世代交代で進化させたAI集団を準備
population_a = PopulationA(20)  # Aの進化した20人のAIプレイヤー
population_b = PopulationB(20)  # Bの進化した20人のAIプレイヤー



# AとBの最適化されたプレイヤーを選択（例として最初のプレイヤーを使用）
player_a = population_a.pop[0]
player_b = population_a.pop[1]
player_c = population_a.pop[2]
player_d = population_a.pop[3]
player_e = population_a.pop[4]

player_f = population_b.pop[0]
player_g = population_b.pop[1]
player_h = population_b.pop[2]
player_i = population_b.pop[3]
player_j = population_b.pop[4]

# ポーカーゲームの設定
config = setup_config(max_round=10, initial_stack=200, small_blind_amount=1)

# AとBの最適化AIを登録
config.register_player(name="AI_A", algorithm=player_a)
config.register_player(name="AI_B", algorithm=player_b)
config.register_player(name="AI_C", algorithm=player_c)
config.register_player(name="AI_D", algorithm=player_d)
config.register_player(name="AI_E", algorithm=player_e)
config.register_player(name="AI_F", algorithm=player_f)
config.register_player(name="AI_G", algorithm=player_g)
config.register_player(name="AI_H", algorithm=player_h)
config.register_player(name="AI_I", algorithm=player_i)
config.register_player(name="AI_J", algorithm=player_j)
# オプション: 追加プレイヤー（ヒューマンまたはスタンダードAI）
#config.register_player(name="AI_C", algorithm=HeuristicPlayer(init_def_prob))

# ゲームを開始して結果を取得
print("Starting the battle between A and B...")
game_result = start_poker(config, verbose=1)
print("Game Result:")
print(game_result)
