import networkx as nx
import matplotlib.pyplot as plt
import math
G = nx.Graph()
# G2= nx.Graph()
G1 = nx.DiGraph()

# G.add_node("p1", position="root")
# G.add_node("a", position="left")
# G.add_node("s1", position="left")
# G.add_node("b", position="right")
# G.add_node("c", position="right")

G.add_node("p1", position="root", numExtention=0)
G.add_node("p1", position="root", numExtention=0)

G.add_node("s1", position="left", numExtention=0)
G.add_node("b", position="right", numExtention=0)
G.add_node("c", position="right", numExtention=0)
G.add_node("a", position="left", numExtention=0)

# G1.add_node("p1", position="root")
# G1.add_node("a", position="right")
# G1.add_node("s1", position="right")
# G1.add_node("b", position="left")
# G1.add_node("c", position="left")

# G.add_edge("p1", "a")
# G.add_edge("p1", "p2")
# G.add_edge("p2", "b")
# G.add_edge("p2", "c")


# G.add_edge("p1", "a")
# G.add_edge("p1", "b")
# G.add_edge("p2", "c")

G.add_edge("p1", "s1")
G.add_edge("p1", "b")
G.add_edge("s1", "c")
G.add_edge("s1", "a")
G.add_edge("a", "s1")


# G1.add_edge("p1", "s1")
# G1.add_edge("p1", "b")
G1.add_edge("s1", "a")
G1.add_edge("s1", "c")

# print G.nodes()
# print G1.nodes()

# print G.successors('p1')

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


# def get_nodes_from_position(G, position=None):
#     return [nodes for nodes, positions in G.nodes(data=True) if positions["position"] == position]




# leaves = [n for n, d in G.out_degree().items() if d == 0]

# print leaves
# print G.edges()
# l =  [(n1, n2) for n1, n2 in G.edges() if (n1 == 's1' or n2 == 's1') and (n2 != 'c')]
# print l
#
# a, b =  [n1 for n1, n2 in l if n1 == 'p1'][0], [n2 for n1, n2 in l if n2 == 'a'][0]
# G.add_edge(a,b)
# for i, j in l:
#     G.remove_edge(i, j)
#
# print G.edges()

# print G.node['c']
# nx.set_edge_attributes(G, 'c', 'd')
mapping = {'c':'d'}
# print G.edges()

G = nx.relabel_nodes(G, mapping)
# print G.nodes(data=True)
# print G.node['d']['position']


# print G.edges()
#
# for i, j in G.edges():
#     print i, j


if not None:
    print "Yes"
else:
    print "No"

# print G.edges()
# print G.node['d']

# print [n for n in G.successors('s1') if n != 'c'][0]

# print get_nodes_from_position(G, 'root')
# G.node['p1']['position'] = 'Root'
# print G.node['p1']['position']
# print math.factorial(2+1)

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

# plt.savefig('BCT_operation_split.png')
# plt.show()



# print G.nodes()
# print G.predecessors("a")
# print nx.is_isomorphic(G, G1)
# i = len([n for n in G.nodes() if n.__contains__('p')])
# print i

