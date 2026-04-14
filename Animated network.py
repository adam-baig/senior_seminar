"""
=============================================================
  ANIMATED TEMPORAL TWO-LAYER SIR SIMULATION
  Senior Seminar
=============================================================

HOW TO RUN:
  1. Open terminal in VS Code  (Ctrl + `)
  2. Install libraries if you haven't yet:
        pip install networkx matplotlib numpy
  3. Run:
        python animated_sir_simulation.py

  A window will pop up with the live animation.
  Press SPACE to pause/resume. Press R to reset.
  Close the window to stop.

WHAT YOU'LL SEE:
  Left panel  — the network, redrawn every time step
                solid blue lines  = household edges (never disappear)
                dashed gray lines = community edges (flicker each step)
                nodes colored by SIR state

  Right panel — the epidemic curve building in real time
                S (blue), I (orange/red), R (green)

PARAMETERS TO PLAY WITH:
  All the numbers you'd want to change are in SECTION 1 below.
  No need to touch anything else to experiment.
=============================================================
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import random
import copy


# ─────────────────────────────────────────────
#  SECTION 1: PARAMETERS  ← change these!
# ─────────────────────────────────────────────

N_NODES        = 0      # number of people (nodes)
HOUSEHOLD_SIZE_MEAN = 4      # average people per household
HOUSEHOLD_SIZE_STD = 0.5
NUM_HOUSEHOLDS = 11     # number of households

BETA           = 0.30    # transmission probability per contact per step
GAMMA          = 0.20    # recovery probability per step

BASE_ACTIVITY  = 0.30    # how often community edges appear (0=never, 1=always)

MAX_STEPS      = 120     # stop after this many steps even if I > 0
ANIMATION_MS   = 400     # milliseconds between frames (lower = faster)
                         # try 200 for fast, 800 for slow and easy to follow

RANDOM_SEED    = None      # change this number to get a different random run
                         # set to None for a different outcome every time


# ─────────────────────────────────────────────
#  SECTION 2: BUILD THE NETWORK
# ─────────────────────────────────────────────

def build_network(seed=None):
    """
    Builds the two-layer network from scratch.
    Called once at startup and again on reset.

    Returns:
      G          — NetworkX graph with household edges + node attributes
      households — dict of {hh_id: [node_ids]}
      pos        — dict of {node_id: (x, y)} for drawing positions
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    G = nx.Graph()
    global N_NODES

    # --- assign households ---
    households = {}
    for i in range(NUM_HOUSEHOLDS):
        for j in range(int(random.normalvariate(HOUSEHOLD_SIZE_MEAN, HOUSEHOLD_SIZE_STD))):
            N_NODES += 1
            G.add_node(N_NODES)
            households.setdefault(i, []).append(N_NODES)
            G.nodes[N_NODES]['household'] = i
            # each node gets its own activity level (personal sociability)
            G.nodes[N_NODES]['activity'] = float(np.clip(
            BASE_ACTIVITY + np.random.normal(0, 0.12), 0.05, 0.95
            ))
            G.nodes[N_NODES]['state'] = 'S'

    # --- patient zero ---
    pz = random.randint(0, N_NODES - 1)
    G.nodes[pz]['state'] = 'I'

    # --- Layer 1: household edges (permanent) ---
    for members in households.values():
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                G.add_edge(members[a], members[b], layer='household')

    # --- node positions: households clustered around a ring ---
    pos = {}
    n_hh = len(households)
    for hh_id, members in households.items():
        hh_angle = (2 * np.pi * hh_id / n_hh) - np.pi / 2
        cx = 2.5 * np.cos(hh_angle)
        cy = 2.5 * np.sin(hh_angle)
        for k, node in enumerate(members):
            local_angle = 2 * np.pi * k / max(len(members), 1)
            pos[node] = (
                cx + 0.45 * np.cos(local_angle),
                cy + 0.45 * np.sin(local_angle)
            )

    return G, households, pos


def build_community_edges(G):
    """
    Layer 2 — generated fresh every time step.
    Each node activates with probability a_i and picks a random contact.
    This edge list is temporary: used once, then thrown away.
    """
    edges = []
    all_nodes = list(G.nodes())
    for node in all_nodes:
        if random.random() < G.nodes[node]['activity']:
            others = [n for n in all_nodes if n != node]
            edges.append((node, random.choice(others)))
    return edges


# ─────────────────────────────────────────────
#  SECTION 3: ONE SIR STEP
# ─────────────────────────────────────────────

def sir_step(G):
    """
    Runs one time step of SIR on the combined two-layer graph.
    Returns the community edges used this step (so we can draw them).
    """
    comm_edges = build_community_edges(G)
    all_edges  = list(G.edges()) + comm_edges   # ← two layers combined

    # collect new states WITHOUT applying yet (synchronous update)
    new_states = {n: G.nodes[n]['state'] for n in G.nodes()}

    # --- infection ---
    for (u, v) in all_edges:
        su, sv = G.nodes[u]['state'], G.nodes[v]['state']
        if su == 'I' and sv == 'S' and random.random() < BETA:
            new_states[v] = 'I'
        elif sv == 'I' and su == 'S' and random.random() < BETA:
            new_states[u] = 'I'

    # --- recovery ---
    for node in G.nodes():
        if G.nodes[node]['state'] == 'I' and random.random() < GAMMA:
            new_states[node] = 'R'

    # apply changes
    for node, state in new_states.items():
        G.nodes[node]['state'] = state

    return comm_edges


# ─────────────────────────────────────────────
#  SECTION 4: DRAWING HELPERS
# ─────────────────────────────────────────────

NODE_COLORS = {'S': '#378ADD', 'I': '#D85A30', 'R': '#1D9E75'}

def get_node_colors(G):
    return [NODE_COLORS[G.nodes[n]['state']] for n in G.nodes()]

def get_counts(G):
    s = sum(1 for n in G.nodes() if G.nodes[n]['state'] == 'S')
    i = sum(1 for n in G.nodes() if G.nodes[n]['state'] == 'I')
    r = sum(1 for n in G.nodes() if G.nodes[n]['state'] == 'R')
    return s, i, r


# ─────────────────────────────────────────────
#  SECTION 5: THE ANIMATED FIGURE
# ─────────────────────────────────────────────

class SIRAnimation:
    """
    Wraps all the matplotlib animation logic in one place.

    matplotlib.animation works by calling an "update" function
    over and over at a fixed interval. Each call:
      1. Runs one SIR step (updates node states)
      2. Clears and redraws the network panel
      3. Adds one new point to the epidemic curve

    The figure has two subplots side by side:
      ax_net  — the network graph (left)
      ax_plot — the epidemic curve (right)
    """

    def __init__(self):
        self.G = None
        self.households = None
        self.pos = None
        self.step = 0
        self.paused = False
        self.done = False

        # history lists for the epidemic curve
        self.s_hist = []
        self.i_hist = []
        self.r_hist = []
        self.comm_edges = []   # current community edges (redrawn each frame)

        # --- set up the figure ---
        self.fig = plt.figure(figsize=(13, 6))
        self.fig.patch.set_facecolor('#F8F8F6')

        # two subplots: network on left, curve on right
        self.ax_net  = self.fig.add_subplot(1, 2, 1)
        self.ax_plot = self.fig.add_subplot(1, 2, 2)

        # title bar with step count and controls hint
        self.title = self.fig.suptitle(
            'Temporal Two-Layer SIR  |  step 0  |  SPACE = pause/resume   R = reset',
            fontsize=11, color='#444441', y=0.98
        )

        # keyboard controls
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        # build initial network
        self._init_sim()

        # --- build legend for network panel ---
        legend_handles = [
            mpatches.Patch(color='#378ADD', label='Susceptible (S)'),
            mpatches.Patch(color='#D85A30', label='Infected (I)'),
            mpatches.Patch(color='#1D9E75', label='Recovered (R)'),
            mlines.Line2D([], [], color='#185FA5', linewidth=2.5,
                          label='household edge (Layer 1 — permanent)'),
            mlines.Line2D([], [], color='#888780', linewidth=1,
                          linestyle='dashed',
                          label='community edge (Layer 2 — temporal)'),
        ]
        self.ax_net.legend(
            handles=legend_handles, loc='upper right',
            fontsize=7.5, framealpha=0.92, edgecolor='#D3D1C7'
        )

        # start the animation
        # interval = ANIMATION_MS between frames
        # blit=False means redraw the whole figure each frame (simpler)
        self.ani = animation.FuncAnimation(
            self.fig,
            self.update,
            interval=ANIMATION_MS,
            blit=False,
            cache_frame_data=False
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    def _init_sim(self):
        """Builds a fresh network and resets all history."""
        self.G, self.households, self.pos = build_network(seed=RANDOM_SEED)
        self.step = 0
        self.done = False
        self.comm_edges = []
        s, i, r = get_counts(self.G)
        self.s_hist = [s]
        self.i_hist = [i]
        self.r_hist = [r]

    def on_key(self, event):
        """Keyboard handler: SPACE toggles pause, R resets."""
        if event.key == ' ':
            self.paused = not self.paused
            state = 'PAUSED' if self.paused else 'running'
            print(f"  [{state}]")
        elif event.key == 'r':
            print("\n  [RESET]\n")
            global N_NODES
            N_NODES = 0
            self._init_sim()
            # clear both axes so the reset draws cleanly
            self.ax_net.cla()
            self.ax_plot.cla()
            self._draw_network()
            self._draw_curve()

    def update(self, frame):
        """
        Called automatically by FuncAnimation every ANIMATION_MS ms.
        This is the core loop of the animation.
        """
        if self.paused or self.done:
            return   # skip this frame without changing anything

        # run one simulation step
        self.comm_edges = sir_step(self.G)
        self.step += 1

        # record counts
        s, i, r = get_counts(self.G)
        self.s_hist.append(s)
        self.i_hist.append(i)
        self.r_hist.append(r)

        # update the title with current step
        self.title.set_text(
            f'Temporal Two-Layer SIR  |  step {self.step}  |  '
            f'S={s}  I={i}  R={r}  |  SPACE = pause   R = reset'
        )

        # redraw both panels
        self.ax_net.cla()
        self.ax_plot.cla()
        self._draw_network()
        self._draw_curve()

        # stop when epidemic is over or max steps reached
        if i == 0 or self.step >= MAX_STEPS:
            self.done = True
            end_msg = (
                f'Epidemic ended at step {self.step}. '
                f'Final: S={s}, R={r} ({round(r/N_NODES*100)}% infected total). '
                f'Press R to run again.'
            )
            self.title.set_text(end_msg)
            print(f"\n  {end_msg}\n")

    def _draw_network(self):
        """
        Draws the network on ax_net.

        Drawing order matters in matplotlib — things drawn later
        appear on top. So we draw edges first, then nodes on top.

        Household edges: solid blue, thick
        Community edges: dashed gray (or orange if touching an infected node)
        Nodes: colored circles with state color
        """
        ax = self.ax_net
        ax.set_facecolor('#F4F3EF')
        ax.set_aspect('equal')
        ax.axis('off')

        # --- draw household edges (Layer 1 — always present) ---
        nx.draw_networkx_edges(
            self.G, self.pos,
            edgelist=list(self.G.edges()),
            edge_color='#185FA5',
            width=2.5,
            alpha=0.55,
            ax=ax,
            style='solid'
        )

        # --- draw community edges (Layer 2 — this step only) ---
        if self.comm_edges:
            # separate infected-adjacent edges so we can color them differently
            hot_edges    = []   # at least one infected endpoint
            normal_edges = []
            for (u, v) in self.comm_edges:
                if (self.G.nodes[u]['state'] == 'I' or
                        self.G.nodes[v]['state'] == 'I'):
                    hot_edges.append((u, v))
                else:
                    normal_edges.append((u, v))

            # build a temporary graph just for drawing community edges
            # (they aren't stored in G permanently)
            tmp = nx.Graph()
            tmp.add_nodes_from(self.G.nodes())
            tmp.add_edges_from(self.comm_edges)

            if normal_edges:
                nx.draw_networkx_edges(
                    tmp, self.pos,
                    edgelist=normal_edges,
                    edge_color='#888780',
                    width=1.0,
                    alpha=0.45,
                    ax=ax,
                    style='dashed'
                )
            if hot_edges:
                # highlight active-infection community edges in orange
                nx.draw_networkx_edges(
                    tmp, self.pos,
                    edgelist=hot_edges,
                    edge_color='#D85A30',
                    width=1.8,
                    alpha=0.7,
                    ax=ax,
                    style='dashed'
                )

        # --- draw nodes ---
        nx.draw_networkx_nodes(
            self.G, self.pos,
            node_color=get_node_colors(self.G),
            node_size=200,
            ax=ax
        )

        # --- draw node id labels ---
        nx.draw_networkx_labels(
            self.G, self.pos,
            font_size=7,
            font_color='white',
            ax=ax
        )

        # --- legend (rebuilt each frame so it survives ax.cla()) ---
        legend_handles = [
            mpatches.Patch(color='#378ADD', label='Susceptible (S)'),
            mpatches.Patch(color='#D85A30', label='Infected (I)'),
            mpatches.Patch(color='#1D9E75', label='Recovered (R)'),
            mlines.Line2D([], [], color='#185FA5', linewidth=2.5,
                          label='household (permanent)'),
            mlines.Line2D([], [], color='#888780', linewidth=1,
                          linestyle='dashed', label='community (temporal)'),
            mlines.Line2D([], [], color='#D85A30', linewidth=1.8,
                          linestyle='dashed', label='active transmission edge'),
        ]
        ax.legend(
            handles=legend_handles,
            loc='upper right', fontsize=7,
            framealpha=0.92, edgecolor='#D3D1C7'
        )
        ax.set_title('network', fontsize=10, pad=6, color='#5F5E5A')

    def _draw_curve(self):
        """
        Draws the live epidemic curve on ax_plot.
        Each call redraws the entire line from the stored history.
        """
        ax = self.ax_plot
        ax.set_facecolor('#F4F3EF')

        steps = list(range(len(self.s_hist)))

        ax.plot(steps, self.s_hist, color='#378ADD', linewidth=2.2,
                label='Susceptible (S)')
        ax.plot(steps, self.i_hist, color='#D85A30', linewidth=2.2,
                label='Infected (I)')
        ax.plot(steps, self.r_hist, color='#1D9E75', linewidth=2.2,
                label='Recovered (R)')

        # mark current values at the rightmost point
        if len(steps) > 0:
            last = steps[-1]
            for val, col in zip(
                [self.s_hist[-1], self.i_hist[-1], self.r_hist[-1]],
                ['#378ADD', '#D85A30', '#1D9E75']
            ):
                ax.plot(last, val, 'o', color=col, markersize=5)

        ax.set_xlim(left=0, right=max(MAX_STEPS, len(steps) + 2))
        ax.set_ylim(bottom=0, top=N_NODES + 1)
        ax.set_xlabel('time step', fontsize=9, color='#5F5E5A')
        ax.set_ylabel('number of nodes', fontsize=9, color='#5F5E5A')
        ax.set_title('epidemic curve', fontsize=10, pad=6, color='#5F5E5A')
        ax.legend(fontsize=8, framealpha=0.92, edgecolor='#D3D1C7')
        ax.grid(True, alpha=0.25, color='#B4B2A9')
        ax.tick_params(labelsize=8, colors='#5F5E5A')
        for spine in ax.spines.values():
            spine.set_edgecolor('#D3D1C7')

        # annotate R0 estimate
        r0 = round(BETA / GAMMA, 2)
        ax.text(
            0.02, 0.97,
            f'β={BETA}  γ={GAMMA}  R₀≈{r0}',
            transform=ax.transAxes,
            fontsize=8, color='#888780',
            verticalalignment='top'
        )


# ─────────────────────────────────────────────
#  SECTION 6: RUN
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 52)
    print("  ANIMATED TEMPORAL TWO-LAYER SIR SIMULATION")
    print("=" * 52)
    print(f"\n  N={N_NODES} nodes   household size average={HOUSEHOLD_SIZE_MEAN}")
    print(f"  β={BETA}   γ={GAMMA}   R₀≈{round(BETA/GAMMA,2)}")
    print(f"  base activity={BASE_ACTIVITY}   seed={RANDOM_SEED}")
    print(f"\n  Controls:")
    print(f"    SPACE  — pause / resume")
    print(f"    R      — reset with a new patient zero")
    print(f"    close the window to exit\n")

    SIRAnimation()