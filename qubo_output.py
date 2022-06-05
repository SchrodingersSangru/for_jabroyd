from dimod import qubo_to_ising
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
import itertools
from dwave.system import DWaveSampler
import random

# Make a small QUBO
class define_qubo:
    
    def __init__(self, qubo_size):
        self.qubo_size = qubo_size
    
    def qubo_1(): 
        
        qubo = {(0,0):1,(1,1):1,(0,1):1}
        return qubo
    
    def qubo_2(self):
        
        # Make a large QUBO
        qubo = {t: random.uniform(-1, 1) for t in itertools.product(range(self.qubo_size), repeat=2)}
        return qubo 

class solver(define_qubo):
    
    def __init__(self, qubo):
        self.qubo = qubo
        

    def solve(self):
        # Set up a composite QPU sampler
        sampler = EmbeddingComposite(DWaveSampler())

        # Solve the small QUBO
        response1 = sampler.sample_qubo(self.qubo)

        # Solve the large QUBO
        
        print(response1)
        
        

if __name__ == "__main__":
    
    qubo_size = 10
    
    qubo = define_qubo(qubo_size).qubo_2()
    
    solution = solver(qubo).solve()
    
