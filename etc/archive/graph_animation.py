import random

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import networkx as nx

# Graph initialization
G = nx.Graph()
G.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8, 9])
G.add_edges_from(
    [
        (1, 2),
        (3, 4),
        (2, 5),
        (4, 5),
        (6, 7),
        (8, 9),
        (4, 7),
        (1, 7),
        (3, 5),
        (2, 7),
        (5, 8),
        (2, 9),
        (5, 7),
    ]
)


# Animation funciton
def animate(i):
    colors = ["r", "b", "g", "y", "w", "m"]
    nx.draw_circular(G, node_color=[random.choice(colors) for j in range(9)])


nx.draw_circular(G)
fig = plt.gcf()

# Animator call
anim = animation.FuncAnimation(fig, animate, frames=20, interval=20, blit=True)
