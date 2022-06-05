from dimod import ConstrainedQuadraticModel, Binary, quicksum
from dwave.system import LeapHybridCQMSampler

# Initialize the CQM solver
sampler = LeapHybridCQMSampler()

# Solve the problem using the CQM solver
sampleset = sampler.sample_cqm(cqm, label='Example - Graph Coloring')
feasible_sampleset = sampleset.filter(lambda row: row.is_feasible)