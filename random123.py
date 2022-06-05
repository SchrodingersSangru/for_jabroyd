from pyqubo import UnaryEncInteger, Array, Placeholder, Constraint
from neal import SimulatedAnnealingSampler
from dwave.system import LeapHybridCQMSampler

# Initialize the CQM solver
class  formulate():
    
    def __init__(self, v, weight, W):
        self.v = v
        self.weight = weight
        self.W = W

    def objective(self):
        x = Array.create('x', shape=len(self.v), vartype='BINARY')

        w_int = UnaryEncInteger('w_int', (0, self.W))

        a = Placeholder('a')

        objective = x.dot(self.v) + a*Constraint(x.dot(self.weight) - w_int, 'weight')**2
        return objective 
    
class get_solution():
    def __init__(self, obj):
        self.obj = obj
        
    def solve(self):
        model = self.obj.compile()
        a = 1.0
        qubo, offset = model.to_qubo(feed_dict={'a': a})
        solution = SimulatedAnnealingSampler().sample_qubo(qubo)
        print(solution)
        # sampler = LeapHybridCQMSampler()

        # # Solve the problem using the CQM solver
        # sampleset = sampler.sample_cqm(self.obj, label='Example - Graph Coloring')
        # feasible_sampleset = sampleset.filter(lambda row: row.is_feasible)
        # print(feasible_sampleset)
        



if __name__ == "__main__":
    
    num_nodes = 8
    v = [5, 8, 3, 2, 8, 4]
    weight = [3, 5, 2, 1, 4, 8]
    W = 12

    obj = formulate(v, weight, W).objective()

    sample = get_solution(obj).solve()
    print(sample)
    # plot_soln(sample, pos)

    