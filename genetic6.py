# 二点交差、評価関数は攻撃率とフィットネス
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
        self.pop = []  # AI players list
        self.size = size  # Number of AI players
        for i in range(size):
            # Generate a random bot
            def_prob = normalize(init_def_prob * (1 + np.random.uniform(-0.25, 0.25, size=(5, 4))))
            self.pop.append(HeuristicPlayer(def_prob, agg=np.random.uniform(0, 2)))

    def compute_weighted_fitness(self):
        """
        Compute selection probabilities by multiplying fitness and aggression.
        """
        base_fitness = self.compute_fitness()  # Calculate current fitness
        weighted_fitness = [base_fitness[i] * self.pop[i].aggression for i in range(self.size)]
        total = sum(weighted_fitness)
        return [f / total for f in weighted_fitness]  # Normalize to probabilities

    def crossover(self, parent1, parent2):
        """
        Two-point crossover to generate a child from two parents.
        """
        point1 = random.randint(1, len(parent1.default_prob) - 2)  # First crossover point
        point2 = random.randint(point1 + 1, len(parent1.default_prob))  # Second crossover point

        child_prob = []
        child_prob.extend(parent1.default_prob[:point1])  # Inherit genes from parent1 up to point1
        child_prob.extend(parent2.default_prob[point1:point2])  # Inherit genes from parent2 between point1 and point2
        child_prob.extend(parent1.default_prob[point2:])  # Inherit remaining genes from parent1

        # Combine aggression values with noise
        child_aggression = (parent1.aggression + parent2.aggression) / 2 + np.random.uniform(-0.1, 0.1)
        return HeuristicPlayer(np.array(child_prob), agg=child_aggression)

    def birth_cycle(self):
        """Generate the next generation considering aggression and fitness."""
        weighted_fitnesses = self.compute_weighted_fitness()
        print("Weighted fitnesses")
        print(weighted_fitnesses)

        # Select parents based on weighted fitness
        new_generation = list(np.random.choice(self.pop, self.size // 2, p=weighted_fitnesses, replace=False))

        # Generate offspring using crossover
        for _ in range(self.size - len(new_generation)):
            parent1, parent2 = np.random.choice(new_generation, 2, replace=False)  # Randomly select two parents
            child = self.crossover(parent1, parent2)  # Create child using two-point crossover
            if random.uniform(0, 1) > 0.75:  # Apply mutation occasionally
                child.mutate()
            new_generation.append(child)

        self.pop = new_generation  # Update population to next generation
        print("New generation created!")

    def compute_fitness(self):
        """
        Divide players into 4 tables and play 5 rounds to calculate fitness.
        """
        total_fitness = [0] * self.size  # Initialize fitness scores
        for rnd in range(5):
            print("Beginning population round {0}".format(rnd))
            tables = np.random.permutation(self.size)  # Shuffle players

            # Divide players into 4 tables
            table1 = [(self.pop[i], i) for i in tables[:self.size // 4]]
            table2 = [(self.pop[i], i) for i in tables[self.size // 4:2 * self.size // 4]]
            table3 = [(self.pop[i], i) for i in tables[2 * self.size // 4:3 * self.size // 4]]
            table4 = [(self.pop[i], i) for i in tables[3 * self.size // 4:]]

            # Calculate fitness for each table
            round_fitness = helper.add([self.play_round(table1), self.play_round(table2), self.play_round(table3), self.play_round(table4)])
            print("The fitness totals for this round are: ", round_fitness)

            # Accumulate fitness scores
            total_fitness = helper.add([round_fitness, total_fitness])
            print(total_fitness)

        return total_fitness

    def play_round(self, players):
        """
        Simulate a poker round with the given players.
        """
        config = setup_config(max_round=20, initial_stack=200, small_blind_amount=1)
        print("Setting up a new table")
        for player, num in players:
            print("Welcoming player {0}".format(num))
            config.register_player(name=num, algorithm=player)

        results = start_poker(config, verbose=0)
        print("The final results of the poker tournament are: ", results)

        fitnesses = [0] * self.size
        for player in results['players']:
            fitnesses[player['name']] = player['stack']  # Record final stack as fitness
        return fitnesses

    def print(self):
        print([(x.default_prob, x.aggression) for x in self.pop])

# Run the genetic algorithm
a = Population(20)
a.print()
for epoch in range(5):
    print("Running epoch {0}".format(epoch))
    a.birth_cycle()
    a.print()
