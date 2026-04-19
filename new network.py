"""
=============================================================
  TEMPORAL TWO-LAYER SIR NETWORK SIMULATION
  Senior Seminar — Math Degree Project
=============================================================

WHAT THIS FILE DOES:
  Models disease spread on a two-layer network using SIR dynamics.
  Layer 1 = household contacts (persistent edges, always present)
  Layer 2 = community contacts (temporal edges, appear/disappear each step)

HOW TO RUN:
  1. Open a terminal in VS Code  (Terminal > New Terminal)
  2. Install required libraries if you haven't yet:
        pip install networkx matplotlib numpy
  3. Run this file:
        python temporal_sir_simulation.py

  Four figures will pop up showing the network and epidemic curve.

LIBRARIES USED:
  networkx  — builds and manages graph objects (nodes + edges)
  matplotlib — draws the graphs and plots
  numpy      — random number generation and math utilities
=============================================================
"""

import networkx as nx          # graph library
import matplotlib.pyplot as plt  # plotting library
import matplotlib.patches as mpatches
import numpy as np             # math / random numbers
import random


# ─────────────────────────────────────────────
#  SECTION 1: PARAMETERS
#  Change these numbers to experiment!
# ─────────────────────────────────────────────

# --- Population ---
N_NODES = 0                 # number of people (nodes)
HOUSEHOLD_SIZE_MEAN = 4     # average people per household
HOUSEHOLD_SIZE_STD = 4/3
NUM_HOUSEHOLDS = 20         # number of households

# --- Disease parameters ---
BETA = 0.30                 # transmission probability per contact per time step
                            # higher = disease spreads more easily (try 0.1 to 0.8)
GAMMA = 0.2                 # recovery probability per time step
                            # higher = people recover faster (try 0.05 to 0.5)

# --- Temporal / activity parameters ---
BASE_ACTIVITY = 0.3         # base probability that any node is "active" (makes
                            # a community contact) on a given time step.
                            # Each node also gets a small random personal modifier.

# --- Simulation length ---
MAX_STEPS = 120             # maximum number of time steps to simulate

# --- Random seed (optional) ---
# Setting a seed makes the simulation reproducible — same result every run.
# Change the number to get a different random outcome, or set to None for
# a fully random run each time.
RANDOM_SEED = None


# ─────────────────────────────────────────────
#  SECTION 2: BUILD THE HOUSEHOLD LAYER
#  This is Layer 1 — the persistent layer.
#  Edges here never disappear.
# ─────────────────────────────────────────────

def build_household_layer(seed=None, household_num=NUM_HOUSEHOLDS, household_avg=HOUSEHOLD_SIZE_MEAN,
                          household_std=HOUSEHOLD_SIZE_STD):
    """
    Creates a graph where every node is connected to everyone
    in their household. These edges are permanent throughout
    the simulation — they represent people who live together
    and are always in contact.

    HOW IT WORKS:
      - We create an empty graph G
      - We add N nodes to G (one per person)
      - We divide them into groups of `household_size`
      - For each group, we add edges between every pair
        (this is called a "clique" or "complete subgraph")

    Returns:
      G        — a NetworkX graph (the household layer)
      households — a dict mapping household_id -> list of node ids
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    G = nx.Graph()          # create an empty undirected graph
    global N_NODES

    households = {}  # will store {household_id: [node_ids]}
    for i in range(household_num):
        # assign each node to a household
        for j in range(int(random.normalvariate(household_avg, household_std))):
            N_NODES += 1
            G.add_node(N_NODES)
            households.setdefault(i, []).append(N_NODES)

    # now add edges within each household
    for hh_id, members in households.items():
        for j in range(len(members)):
            for k in range(j + 1, len(members)):
                # add a permanent edge between every pair in the household
                G.add_edge(members[j], members[k], layer='household')

    print(f"Household layer built:")
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} household edges")
    print(f"  {len(households)} households of size {household_num}\n")

    return G, households


# ─────────────────────────────────────────────
#  SECTION 3: ASSIGN ACTIVITY POTENTIALS
#  This gives each person their own "sociability"
#  score — how likely they are to make community
#  contacts on any given day.
# ─────────────────────────────────────────────

def assign_activity_potentials(G, base_activity, seed=None):
    """
    Assigns each node an activity potential a_i in [0, 1].

    Think of a_i as how socially active that person is.
    We draw from a bounded distribution so some people are
    more active (higher a_i) and some less active (lower a_i).
    This creates natural heterogeneity — not everyone has the
    same number of contacts, which is more realistic.

    The value is stored directly on each node in the graph
    using NetworkX's node attribute system: G.nodes[i]['activity']
    """
    if seed is not None:
        np.random.seed(seed)

    for node in G.nodes():
        # draw activity from a distribution centered on base_activity
        # clip to [0.05, 0.95] so no one is completely isolated or
        # always-on
        raw = base_activity + np.random.normal(0, 0.15)
        G.nodes[node]['activity'] = float(np.clip(raw, 0.05, 0.95))

    return G


# ─────────────────────────────────────────────
#  SECTION 4: INITIALIZE SIR STATES
#  Every node starts as Susceptible except for
#  one "patient zero" who starts as Infected.
# ─────────────────────────────────────────────

def initialize_sir_states(G, initial_infected=1, seed=None):
    """
    Sets the initial disease state for every node.

    States:
      'S' = Susceptible  (healthy, can catch the disease)
      'I' = Infected     (sick and contagious)
      'R' = Recovered    (immune, cannot catch or spread)

    We set all nodes to 'S' first, then randomly choose
    `initial_infected` nodes to be 'I' (patient zeros).
    """
    if seed is not None:
        random.seed(seed)

    # set everyone to susceptible first
    for node in G.nodes():
        G.nodes[node]['state'] = 'S'

    # pick random patient zero(s)
    patient_zeros = random.sample(list(G.nodes()), initial_infected)
    for node in patient_zeros:
        G.nodes[node]['state'] = 'I'

    print(f"SIR states initialized:")
    print(f"  Patient zero(s): nodes {patient_zeros}\n")

    return G, patient_zeros


# ─────────────────────────────────────────────
#  SECTION 5: BUILD COMMUNITY EDGES (TEMPORAL)
#  This is Layer 2 — rebuilt fresh every time step.
#  This is the "temporal" part of your model.
# ─────────────────────────────────────────────

def build_community_edges(G):
    """
    At each time step, each node independently decides whether
    to "activate" and make a community contact.

    This is the activity-driven model:
      - For each node i, draw r ~ Uniform(0, 1)
      - If r < a_i (the node's activity potential), node i is active
      - Active node i picks one random other node j
      - Temporarily add edge (i, j) to the graph

    These edges are NOT stored permanently — the function returns
    them as a separate list, and they are only used for this one
    time step. After infection is checked, they are discarded.

    Returns:
      community_edges — list of (node_i, node_j) tuples
    """
    community_edges = []
    all_nodes = list(G.nodes())

    for node in all_nodes:
        a_i = G.nodes[node]['activity']  # this node's activity potential

        if random.random() < a_i:  # node activates with probability a_i
            # pick a random contact from the rest of the population
            others = [n for n in all_nodes if n != node]
            contact = random.choice(others)
            community_edges.append((node, contact))

    return community_edges


# ─────────────────────────────────────────────
#  SECTION 6: ONE TIME STEP OF SIR DYNAMICS
#  This is where the disease actually spreads.
#  We combine both layers and check every edge.
# ─────────────────────────────────────────────

def sir_step(G, household_edges_set, beta, gamma):
    """
    Runs one time step of SIR dynamics on the combined two-layer graph.

    HOW THE TWO LAYERS ARE COMBINED:
      The "current graph" at time t is:
        permanent household edges   (always present)
      + temporary community edges   (generated fresh this step)

      We check infection across BOTH sets of edges together.
      This is what makes it a two-layer model.

    INFECTION LOGIC:
      For each edge (i, j) in the combined graph:
        if one endpoint is 'I' and the other is 'S':
          the 'S' node becomes 'I' with probability beta

    RECOVERY LOGIC:
      For each node in state 'I':
        move to 'R' with probability gamma

    WHY WE COLLECT NEW_STATES SEPARATELY:
      If we updated states immediately as we went, an infection from
      node A could make node B newly infected, and then in the same
      loop that newly infected B could infect C — all in one step.
      That's not realistic. Instead we collect all changes first,
      then apply them all at once at the end. This is called a
      "synchronous update."

    Returns:
      new_I  — list of newly infected nodes this step
      new_R  — list of newly recovered nodes this step
    """

    # Step 1: build the community layer for this time step
    community_edges = build_community_edges(G)

    # Step 2: combine household + community edges into one edge list
    # household edges come from the graph G itself
    # community edges are the fresh list we just built
    all_edges = list(G.edges()) + community_edges

    # Step 3: collect state changes (don't apply yet)
    new_states = {node: G.nodes[node]['state'] for node in G.nodes()}

    new_I = []  # track who gets newly infected this step
    new_R = []  # track who recovers this step

    # --- INFECTION: check every edge ---
    for (u, v) in all_edges:
        state_u = G.nodes[u]['state']
        state_v = G.nodes[v]['state']

        # case 1: u is infected, v is susceptible
        if state_u == 'I' and state_v == 'S':
            if random.random() < beta:      # flip coin weighted by beta
                new_states[v] = 'I'         # v gets infected
                new_I.append(v)

        # case 2: v is infected, u is susceptible
        elif state_v == 'I' and state_u == 'S':
            if random.random() < beta:
                new_states[u] = 'I'
                new_I.append(u)

    # --- RECOVERY: check every infected node ---
    for node in G.nodes():
        if G.nodes[node]['state'] == 'I':
            if random.random() < gamma:     # flip coin weighted by gamma
                new_states[node] = 'R'      # node recovers
                new_R.append(node)

    # Step 4: apply all state changes at once (synchronous update)
    for node, state in new_states.items():
        G.nodes[node]['state'] = state

    return new_I, new_R, community_edges


# ─────────────────────────────────────────────
#  SECTION 7: RUN THE FULL SIMULATION
#  Loop over time steps, record counts at each step.
# ─────────────────────────────────────────────

def run_simulation(G, beta, gamma, max_steps):
    """
    Runs the full SIR simulation for up to `max_steps` steps,
    or until there are no more infected nodes (epidemic over).

    At each step we:
      1. Run one SIR step (sir_step function above)
      2. Count how many nodes are in each state
      3. Record those counts in our history lists

    Returns:
      history — dict with lists of S, I, R counts per step
      last_community_edges — community edges from final step (for plotting)
    """

    # count initial states before any steps
    s_counts = [sum(1 for n in G.nodes() if G.nodes[n]['state'] == 'S')]
    i_counts = [sum(1 for n in G.nodes() if G.nodes[n]['state'] == 'I')]
    r_counts = [sum(1 for n in G.nodes() if G.nodes[n]['state'] == 'R')]

    household_edges_set = set(G.edges())  # store original edges for reference
    last_community_edges = []

    print("Running simulation...")
    print(f"  Step 0: S={s_counts[0]}, I={i_counts[0]}, R={r_counts[0]}")

    for t in range(1, max_steps + 1):
        # run one time step
        new_I, new_R, community_edges = sir_step(G, household_edges_set, beta, gamma)
        last_community_edges = community_edges

        # count states
        s = sum(1 for n in G.nodes() if G.nodes[n]['state'] == 'S')
        i = sum(1 for n in G.nodes() if G.nodes[n]['state'] == 'I')
        r = sum(1 for n in G.nodes() if G.nodes[n]['state'] == 'R')

        s_counts.append(s)
        i_counts.append(i)
        r_counts.append(r)

        # print progress every 10 steps
        if t % 10 == 0 or i == 0:
            print(f"  Step {t}: S={s}, I={i}, R={r}  (+{len(new_I)} infected, +{len(new_R)} recovered)")

        # stop early if epidemic is over
        if i == 0:
            print(f"\nEpidemic ended at step {t}.")
            print(f"Final attack rate: {r}/{N_NODES} = {r/N_NODES:.1%} of population infected\n")
            break

    history = {
        'S': s_counts,
        'I': i_counts,
        'R': r_counts,
        'steps': list(range(len(s_counts)))
    }

    return history, last_community_edges


# ─────────────────────────────────────────────
#  SECTION 8: VISUALIZATION
#  Draw the network and epidemic curve.
# ─────────────────────────────────────────────

def get_node_colors(G):
    """Returns a color for each node based on its SIR state."""
    color_map = {'S': '#378ADD', 'I': '#D85A30', 'R': '#1D9E75'}
    return [color_map[G.nodes[n]['state']] for n in G.nodes()]


def build_layout(G, households, n_nodes):
    """
    Creates a layout where household members are clustered together.
    Each household is positioned around a circle, and members of
    the same household are placed close together.
    """
    pos = {}
    n_households = len(households)

    for hh_id, members in households.items():
        # position each household around a big circle
        hh_angle = (2 * np.pi * hh_id / n_households) - np.pi / 2
        hh_radius = 2.5
        hh_cx = hh_radius * np.cos(hh_angle)
        hh_cy = hh_radius * np.sin(hh_angle)

        # position members in a small circle within the household cluster
        for k, node in enumerate(members):
            member_angle = 2 * np.pi * k / max(len(members), 1)
            member_radius = 0.4
            pos[node] = (
                hh_cx + member_radius * np.cos(member_angle),
                hh_cy + member_radius * np.sin(member_angle)
            )
    return pos


def draw_network_snapshot(G, households, community_edges, title, ax):
    """
    Draws the network at a single point in time, showing:
      - Nodes colored by SIR state
      - Household edges as solid blue lines (Layer 1)
      - Community edges as dashed gray lines (Layer 2)
    """
    pos = build_layout(G, households, G.number_of_nodes())
    node_colors = get_node_colors(G)

    household_edge_list = list(G.edges())

    # draw household edges (solid, blue) — Layer 1
    nx.draw_networkx_edges(
        G, pos, edgelist=household_edge_list,
        edge_color='#185FA5', width=2.0, alpha=0.6, ax=ax,
        style='solid'
    )

    # draw community edges (dashed, gray) — Layer 2
    # community_edges is a list of tuples, not in G.edges() format,
    # so we create a temporary graph just for drawing them
    if community_edges:
        temp_G = nx.Graph()
        temp_G.add_nodes_from(G.nodes())
        temp_G.add_edges_from(community_edges)
        valid_comm = [(u, v) for (u, v) in community_edges if u in pos and v in pos]
        nx.draw_networkx_edges(
            temp_G, pos, edgelist=valid_comm,
            edge_color='#888780', width=1.0, alpha=0.5, ax=ax,
            style='dashed'
        )

    # draw nodes
    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors,
        node_size=180, ax=ax
    )

    # draw node labels (node id numbers)
    nx.draw_networkx_labels(
        G, pos, font_size=7, font_color='white', ax=ax
    )

    # legend
    legend_handles = [
        mpatches.Patch(color='#378ADD', label='Susceptible (S)'),
        mpatches.Patch(color='#D85A30', label='Infected (I)'),
        mpatches.Patch(color='#1D9E75', label='Recovered (R)'),
        plt.Line2D([0], [0], color='#185FA5', linewidth=2, label='Household edge (Layer 1)'),
        plt.Line2D([0], [0], color='#888780', linewidth=1.5, linestyle='dashed', label='Community edge (Layer 2)'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=7, framealpha=0.9)
    ax.set_title(title, fontsize=11, pad=8)
    ax.axis('off')


def plot_epidemic_curve(history, beta, gamma, ax):
    """
    Plots the classic SIR epidemic curve — S, I, R counts over time.
    The 'hump' shape of the I curve is what epidemiologists call
    the epidemic curve.
    """
    steps = history['steps']

    ax.plot(steps, history['S'], color='#378ADD', linewidth=2.5, label='Susceptible (S)')
    ax.plot(steps, history['I'], color='#D85A30', linewidth=2.5, label='Infected (I)')
    ax.plot(steps, history['R'], color='#1D9E75', linewidth=2.5, label='Recovered (R)')

    # find and mark the epidemic peak
    peak_step = np.argmax(history['I'])
    peak_val  = history['I'][peak_step]
    ax.axvline(x=peak_step, color='#D85A30', linestyle=':', alpha=0.5)
    ax.annotate(
        f'peak: {peak_val} infected\n(step {peak_step})',
        xy=(peak_step, peak_val),
        xytext=(peak_step + 2, peak_val - 1),
        fontsize=8, color='#D85A30'
    )

    ax.set_xlabel('Time step', fontsize=10)
    ax.set_ylabel('Number of nodes', fontsize=10)
    ax.set_title(
        f'Epidemic curve  (β={beta}, γ={gamma}, R₀≈{beta/gamma:.2f})',
        fontsize=11
    )
    ax.legend(fontsize=9)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    # note: R0 here is an approximation. the true epidemic threshold
    # on this network also depends on the degree distribution.


def plot_activity_distribution(G, ax):
    """
    Shows the distribution of activity potentials across all nodes.
    This is the F(a) distribution from the activity-driven model.
    A wider spread means more heterogeneity in contact behavior.
    """
    activities = [G.nodes[n]['activity'] for n in G.nodes()]
    ax.hist(activities, bins=10, color='#185FA5', alpha=0.7, edgecolor='white')
    ax.axvline(np.mean(activities), color='#D85A30', linestyle='--',
               label=f'mean = {np.mean(activities):.2f}')
    ax.set_xlabel('Activity potential aᵢ', fontsize=10)
    ax.set_ylabel('Number of nodes', fontsize=10)
    ax.set_title('Distribution of activity potentials F(a)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


# ─────────────────────────────────────────────
#  SECTION 9: EXPERIMENT RUNNER
#  Run multiple simulations varying a parameter
#  and compare the epidemic curves. This is what
#  connects to your research question about how
#  graph structure affects disease spread.
# ─────────────────────────────────────────────

def run_activity_experiment():
    """
    Runs the simulation multiple times with different activity levels
    and overlays the epidemic curves. This shows how the temporal
    component (how often community edges appear) affects spread.

    This is a simple version of your 'meta-random model' comparison:
    you're varying the edge-activation distribution F(a) and
    observing how the epidemic curve changes.
    """
    print("\n" + "="*50)
    print("EXPERIMENT: varying activity level")
    print("="*50)

    activity_levels = [0.2, 0.4, 0.6, 0.8]
    fig, ax = plt.subplots(figsize=(9, 5))

    colors = ['#185FA5', '#1D9E75', '#D85A30', '#7F77DD']

    for activity, color in zip(activity_levels, colors):
        # build fresh graph for each experiment
        G_exp, hh_exp = build_household_layer(seed=RANDOM_SEED)
        G_exp = assign_activity_potentials(G_exp, activity, seed=RANDOM_SEED)
        G_exp, _ = initialize_sir_states(G_exp, initial_infected=1, seed=RANDOM_SEED)

        hist, _ = run_simulation(G_exp, BETA, GAMMA, MAX_STEPS)
        ax.plot(hist['steps'], hist['I'], color=color, linewidth=2,
                label=f'activity = {activity}')

    ax.set_xlabel('Time step', fontsize=10)
    ax.set_ylabel('Infected nodes (I)', fontsize=10)
    ax.set_title('Effect of activity level on epidemic curve\n(higher activity = more community edges per step)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig('experiment_activity.png', dpi=150)
    print("\nExperiment figure saved as 'experiment_activity.png'")
    plt.show()


# ─────────────────────────────────────────────
#  SECTION 10: MAIN — runs everything
# ─────────────────────────────────────────────

def main():
    print("="*50)
    print("  TEMPORAL TWO-LAYER SIR SIMULATION")
    print("="*50 + "\n")
    print(f"Parameters:")
    print(f"  N = {N_NODES} nodes, number of households = {NUM_HOUSEHOLDS}")
    print(f"  β = {BETA}, γ = {GAMMA}, R₀ ≈ {BETA/GAMMA:.2f}")
    print(f"  base activity = {BASE_ACTIVITY}\n")

    # --- Step 1: build the two-layer graph ---
    G, households = build_household_layer(seed=RANDOM_SEED)
    G = assign_activity_potentials(G, BASE_ACTIVITY, seed=RANDOM_SEED)
    G, patient_zeros = initialize_sir_states(G, initial_infected=1, seed=RANDOM_SEED)

    # save a snapshot of the initial network for plotting later
    import copy
    G_initial = copy.deepcopy(G)

    # --- Step 2: run the simulation ---
    history, last_community_edges = run_simulation(G, BETA, GAMMA, MAX_STEPS)

    # --- Step 3: draw everything ---
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('Temporal Two-Layer SIR Network Simulation', fontsize=14, y=0.98)

    # top left: initial network state
    ax1 = fig.add_subplot(2, 2, 1)
    initial_comm = build_community_edges(G_initial)  # sample community edges for t=0
    draw_network_snapshot(G_initial, households, initial_comm,
                          'Network at t=0\n(patient zero in orange)', ax1)

    # top right: final network state
    ax2 = fig.add_subplot(2, 2, 2)
    draw_network_snapshot(G, households, last_community_edges,
                          'Network at end of simulation', ax2)

    # bottom left: epidemic curve
    ax3 = fig.add_subplot(2, 2, 3)
    plot_epidemic_curve(history, BETA, GAMMA, ax3)

    # bottom right: activity distribution
    ax4 = fig.add_subplot(2, 2, 4)
    plot_activity_distribution(G_initial, ax4)

    plt.tight_layout()
    plt.savefig('sir_simulation.png', dpi=150, bbox_inches='tight')
    print("Main figure saved as 'sir_simulation.png'")
    plt.show()

    # --- Step 4: run the activity experiment ---
    run_activity_experiment()


# this block runs main() only when you execute this file directly.
# if you import this file from another script, main() won't auto-run.
if __name__ == '__main__':
    main()