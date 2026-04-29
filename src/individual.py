import numpy as np

class Individual:
    def __init__(self, genotype_length=None, genotype=None):
        if genotype is not None:
            self.genotype = np.array(genotype)
        else:
            # 0: Left, 1: Down, 2: Right, 3: Up
            self.genotype = np.random.randint(0, 4, size=genotype_length)
        
        # initialize fitness with -inf to ensure any valid fitness is better
        self.fitness = -float('inf')
        self.success = 0
        self.steps = 0

    def evaluate(self, env, weights):
        self.fitness, self.success, self.steps = env.evaluate_policy(self.genotype, weights)

    def clone(self):
        ind = Individual(genotype=self.genotype.copy())
        ind.fitness = self.fitness
        ind.success = self.success
        ind.steps = self.steps
        return ind
