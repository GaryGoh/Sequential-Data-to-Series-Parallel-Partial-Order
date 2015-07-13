__author__ = 'GaryGoh'

import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt


B = nx.Graph()
B.add_nodes_from([1,2,3,4], bipartite=0) # Add the node attribute "bipartite"
B.add_nodes_from(['a','b','c'], bipartite=1)
B.add_edges_from([(1,'a'), (1,'b'), (2,'b'), (2,'c'), (3,'c'), (4,'a')])
for i in B:
    print i


# # same layout using matplotlib with no labels
plt.title("draw_networkx")
pos = nx.graphviz_layout(B, prog='dot')
nx.draw(B, pos, with_labels=True, arrows=False)
# nx.draw(G1,pos,with_labels=True,arrows=False)

# plt.savefig('nx_test.png')
plt.show()