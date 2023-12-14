from networkx import graph


def get_dummy_graph():
    g = graph.Graph()
    g.add_node(0, atomic_number=26)
    # N_2
    g.add_node(1, atomic_number=7)
    g.add_node(5, atomic_number=7)
    # HO^-
    g.add_node(2, atomic_number=8)
    g.add_node(6, atomic_number=1)
    # Cl
    g.add_node(3, atomic_number=17)
    # HO^-
    g.add_node(4, atomic_number=8)
    g.add_node(7, atomic_number=1)
    g.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (2, 6), (4, 7)])
    return g
