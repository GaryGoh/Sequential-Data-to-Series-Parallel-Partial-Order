import networkx as nx
import matplotlib.pyplot as plt
import math
G = nx.DiGraph()
G1 = nx.DiGraph()

# G.add_node("p1", position="root")
# G.add_node("a", position="left")
# G.add_node("s1", position="left")
# G.add_node("b", position="right")
# G.add_node("c", position="right")

G.add_node("p1", position="root", numExtention=0)
G.add_node("a", position="left", numExtention=0)
G.add_node("s1", position="left", numExtention=0)
G.add_node("b", position="right", numExtention=0)
G.add_node("c", position="right", numExtention=0)

# G.add_node("p1", position="root")
# G.add_node("a", position="right")
# G.add_node("s1", position="right")
# G.add_node("b", position="left")
# G.add_node("c", position="left")

# G.add_edge("p1", "a")
# G.add_edge("p1", "p2")
# G.add_edge("p2", "b")
# G.add_edge("p2", "c")


# G.add_edge("p1", "a")
# G.add_edge("p1", "b")
# G.add_edge("p2", "c")

G.add_edge("p1", "s1")
G.add_edge("p1", "b")
G.add_edge("s1", "a")
G.add_edge("s1", "c")

# G1.add_edge("p1", "s1")
# G1.add_edge("p1", "b")
# G1.add_edge("s1", "a")
# G1.add_edge("s1", "c")



# for i in G:
# print i
# print "**********"
# for i in G1:
# print i

# print G.nodes(data=True)
# print list(nx.dfs_postorder_nodes(G,"p1"))

# try:
#     print list(nx.dfs_postorder_nodes(G, "p"))
# except Exception:
#     raise TypeError("There is no this node in the binary construction tree")
# print len(G.nodes())


# print G.node["root"]
# print G.nodes(data=True)


def get_nodes_from_position(G, position=None):
    return [nodes for nodes, positions in G.nodes(data=True) if positions["position"] == position]



# print get_nodes_from_position(G, 'root')
# G.node['p1']['position'] = 'Root'
# print G.node['p1']['position']
print math.factorial(2+1)

# print nx.info(G)

# write dot file to use with graphviz
# run "dot -Tpng test.dot >test.png"
# nx.write_dot(G,'test.dot')
#
# # same layout using matplotlib with no labels
# plt.title("draw_networkx")
# pos = nx.graphviz_layout(G1, prog='dot')
# nx.draw_graphviz(G, prog='dot', with_labels=True, datas=True)
# nx.draw(G1, pos, with_labels=True, arrows=False)
# nx.draw(G1,pos,with_labels=True,arrows=False)

# plt.savefig('nx_test.png')
# plt.show()



# print G.nodes()
# print G.predecessors("a")
# print nx.is_isomorphic(G, G1)

