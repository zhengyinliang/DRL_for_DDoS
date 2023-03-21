import networkx as nx
G = nx.Graph()

#G.add_edge()
G.add_edge(1, 2, weight=3)
G.add_edge(1, 3, weight=7, cs=15, length=342.7)
G[1][3]['cs']  += 10
print(G[1][3]['cs'])


slot = G[1][3]['cs']
slot = 1
print(G[1][3]['cs'])
