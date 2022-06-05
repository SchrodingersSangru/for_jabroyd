from pyqubo import UnaryEncInteger, Array, Placeholder, Constraint
from neal import SimulatedAnnealingSampler

class  solver():
    
    def __init__(self, v, weight, W):
        self.v = v
        self.weight = weight
        self.W = W

    def solve(self):
        x = Array.create('x', shape=len(self.v), vartype='BINARY')

        w_int = UnaryEncInteger('w_int', (0, self.W))

        a = Placeholder('a')

        H = x.dot(self.v) + a*Constraint(x.dot(self.weight) - w_int, 'weight')**2

        model = H.compile()
        a = 1.0
        qubo, offset = model.to_qubo(feed_dict={'a': a})
        # print(qubo)
        sol = SimulatedAnnealingSampler().sample_qubo(qubo)
        print(sol)