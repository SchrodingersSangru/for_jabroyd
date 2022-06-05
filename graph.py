import matplotlib
import networkx as nx
from dimod import ConstrainedQuadraticModel, Binary, quicksum
from dwave.system import LeapHybridCQMSampler

try:
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib.use("agg")
    import matplotlib.pyplot as plt

def build_graph(num_nodes):
    """Build graph."""

    print("\nBuilding graph...")

    G = nx.powerlaw_cluster_graph(num_nodes, 3, 0.4)
    pos = nx.spring_layout(G)
    nx.draw(G, pos=pos, node_size=50, edgecolors='k', cmap='hsv')
    plt.savefig("original_graph.png")

    return G, pos

def build_cqm(G, num_colors):
    """Build CQM model."""

    print("\nBuilding constrained quadratic model...")

    # Initialize the CQM object
    cqm = ConstrainedQuadraticModel()

    # Build CQM variables
    colors = {n: {c: Binary((n, c)) for c in range(num_colors)} for n in G.nodes}

    # Add constraint to make variables discrete
    for n in G.nodes():
        cqm.add_discrete([(n, c) for c in range(num_colors)])
  
    # Build the constraints: edges have different color end points
    for u, v in G.edges:
        for c in range(num_colors):
            cqm.add_constraint(colors[u][c]*colors[v][c] == 0)

    return cqm

def run_hybrid_solver(cqm):
    """Solve CQM using hybrid solver."""

    print("\nRunning hybrid sampler...")

    # Initialize the CQM solver
    sampler = LeapHybridCQMSampler()

    # Solve the problem using the CQM solver
    sampleset = sampler.sample_cqm(cqm, label='Example - Graph Coloring')
    feasible_sampleset = sampleset.filter(lambda row: row.is_feasible)

    try:
        sample = feasible_sampleset.first.sample
    except:
        print("\nNo feasible solutions found.")
        exit()

    soln = {key[0]: key[1] for key, val in sample.items() if val == 1.0}

    return soln

def plot_soln(sample, pos):
    """Plot results and save file.
    
    Args:
        sample (dict):
            Sample containing a solution. Each key is a node and each value 
            is an int representing the node's color.
        pos (dict):
            Plotting information for graph so that same graph shape is used.
    """

    print("\nProcessing sample...")

    node_colors = [sample[i] for i in G.nodes()]
    nx.draw(G, pos=pos, node_color=node_colors, node_size=50, edgecolors='k', cmap='hsv')
    fname = 'graph_result.png'
    plt.savefig(fname)

    print("\nSaving results in {}...".format(fname))

# ------- Main program -------
if __name__ == "__main__":

    num_nodes = 8

    G, pos = build_graph(num_nodes)
    num_colors = max(d for _, d in G.degree()) + 1

    cqm = build_cqm(G, num_colors)

    sample = run_hybrid_solver(cqm)

    plot_soln(sample, pos)

    colors_used = max(sample.values())+1
    print("\nColors used:", colors_used, "\n")