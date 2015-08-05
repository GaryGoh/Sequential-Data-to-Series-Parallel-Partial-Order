import networkx as nx
import matplotlib.pyplot as plt
import math

G = nx.DiGraph()
G2 = nx.DiGraph()
G3 = nx.DiGraph()

G1 = nx.DiGraph()

G.add_node("p1", position="root")
G.add_node("s1", position="left")
G.add_node("c", position="right")
G.add_node("a", position="left")
G.add_node("b", position="right")

G2.add_node("p", position="root")
G2.add_node("s1", position="left")
G2.add_node("c", position="right")
G2.add_node("a", position="left")
G2.add_node("b", position="right")

G3.add_node("p1", position="root")
G3.add_node("p2", position="left")
G3.add_node("b", position="right")
G3.add_node("a", position="left")
G3.add_node("c", position="right")

# G.add_node("p1", position="root", numExtention=0)
# G.add_node("s1", position="left", numExtention=0)
# G.add_node("b", position="right", numExtention=0)
# G.add_node("c", position="right", numExtention=0)
# G.add_node("a", position="left", numExtention=0)
# G.add_node("e", position="right", numExtention=0)
# G.add_node("d", position="left", numExtention=0)
# G.add_node("f", position="right", numExtention=0)
# G.add_node("g", position="left", numExtention=0)
# G.add_node("h", position="right", numExtention=0)
# G.add_node("i", position="left", numExtention=0)

G1.add_node("p1", position="root")
G1.add_node("a", position="right")
G1.add_node("s1", position="right")
G1.add_node("b", position="left")
G1.add_node("c", position="left")
G1.add_node("d", position="left")
G1.add_node("e", position="left")
G1.add_node("f", position="left")
G1.add_node("g", position="left")
G1.add_node("h", position="left")
G1.add_node("i", position="left")
G1.add_node("j", position="left")
G1.add_node("k", position="left")

G.add_edge("p1", "c")
G.add_edge("p1", "s1")
G.add_edge("s1", "a")
G.add_edge("s1", "b")

G2.add_edge("p", "c")
G2.add_edge("p", "s1")
G2.add_edge("s1", "a")
G2.add_edge("s1", "b")

G3.add_edge("p1", "p2")
G3.add_edge("p1", "b")
G3.add_edge("p2", "a")
G3.add_edge("p2", "c")

# G.add_edge("p1", "a")
# G.add_edge("p1", "b")
# G.add_edge("p2", "c")

# G.add_edge("p1", "s1")
# G.add_edge("p1", "b")
# G.add_edge("s1", "a")
# G.add_edge("s1", "c")
# G.add_edge("c", "d")
# G.add_edge("c", "e")
# G.add_edge("b", "f")
# G.add_edge("b", "g")
# G.add_edge("e", "h")
# G.add_edge("e", "i")

G1.add_edge("p1", "a")
G1.add_edge("p1", "s1")
G1.add_edge("s1", "b")
G1.add_edge("s1", "c")
G1.add_edge("b", "d")
G1.add_edge("b", "e")
G1.add_edge("d", "f")
G1.add_edge("d", "g")
G1.add_edge("f", "h")
G1.add_edge("f", "i")
G1.add_edge("c", "j")
G1.add_edge("c", "k")
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
# print list(nx.dfs_postorder_nodes(G, "p"))
# except Exception:
# raise TypeError("There is no this node in the binary construction tree")
# print len(G.nodes())


# print G.node["root"]
# print G.nodes(data=True)


# def get_nodes_from_position(G, position=None):
# return [nodes for nodes, positions in G.nodes(data=True) if positions["position"] == position]




# leaves = [n for n, d in G.out_degree().items() if d == 0]

# print leaves
# print G.edges()
# l =  [(n1, n2) for n1, n2 in G.edges() if (n1 == 's1' or n2 == 's1') and (n2 != 'c')]
# print l
#
# a, b =  [n1 for n1, n2 in l if n1 == 'p1'][0], [n2 for n1, n2 in l if n2 == 'a'][0]
# G.add_edge(a,b)
# for i, j in l:
# G.remove_edge(i, j)
#
# print G.edges()

# print G.node['c']
# nx.set_edge_attributes(G, 'c', 'd')
# mapping = {'c':'d'}
# print G.edges()

# G = nx.relabel_nodes(G, mapping)
# print G.nodes(data=True)
# print G.node['d']['position']

# print list(nx.dfs_edges(G))
# print list(nx.dfs_edges(G1))

# print nx.dfs_tree(G, 'p1').edges()
# print nx.dfs_tree(G1, 'p1').edges()

def dfs_inorder(G, source=None):
    # post = (v for u, v, d in nx.dfs_labeled_edges(G, source=source)
    #         if d['dir'] == 'reverse')

    for u, v, d in nx.dfs_labeled_edges(G, source=source):
        if d['dir'] == 'reverse':
            print u, v, d

    print "**************************************"
    for u, v, d in nx.dfs_labeled_edges(G, source=source):
        if d['dir'] == 'forward':
            print u, v, d

    print "**************************************"
    for u, v, d in nx.dfs_labeled_edges(G, source=source):
        if d['dir'] == 'nontree':
            print u, v, d
            # print post

            # return post


# print list(nx.dfs_postorder_nodes(G))
# print list(nx.dfs_preorder_nodes(G))

# print list(nx.dfs_postorder_nodes(G1))

# print list(nx.bfs_edges(G1, 'p1'))
# print list(nx.dfs_postorder_nodes(G1, 'p1'))
# print G.predecessors('s1')
# print [n for n, d in G1.degree().items() if d == 1]
# for u, v, d in nx.dfs_labeled_edges(G):
#     print u, v, d

# G2 = G1.reverse()
# print list(nx.dfs_postorder_nodes(G1))

# dfs_inorder(G1)


# print G.successors('p1')[0]
# print G1.out_degree('a')
# print G.edges()
#
# for i, j in G.edges():
#     print i, j


# if not None:
#     print "Yes"
# else:
#     print "No"

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
# print list(nx.dfs_edges(G1, 'p1'))
bfs = list(nx.dfs_edges(G1, 'p1'))

# n100 = [i for i, j in bfs]
n100 = [j for i, j in bfs if G1.out_degree(j) == 0]

output = []
for i in n100:
    if i not in output:
       output.append(i)
# print output

# print G.predecessors()
# plt.title("draw_networkx")
# pos = nx.graphviz_layout(G, prog='dot')
# nx.draw_graphviz(G1, prog='dot', with_labels=True, datas=True)
# nx.draw(G, pos, with_labels=True, arrows=False)
# nx.draw_graphviz(G, prog='dot', with_labels=True, datas=True)

# nx.draw(G1,pos,with_labels=True,arrows=False)

# plt.savefig('BCT_operation_split.png')
# plt.show()

# left, right = G.successors('p1')
# print left, right, G.successors('p1')

# print G.nodes()
# print G.nodes()
# print G.predecessors("a")
# print nx.is_isomorphic(G, G1)
# i = len([n for n in G.nodes() if n.__contains__('p')])
# print i

#
import networkx.algorithms.isomorphism as iso
#
# position_category = iso.categorical_node_match(G.node['p1'], G2.node['p'])
position_category = iso.categorical_node_match('position', ['left', 'right'])

# # position_category = iso.categorical_edge_match('left', 'right')
#
#
#

# GM = iso.GraphMatcher(G,G3)
# print GM.is_isomorphic()
#
#
# print G3.nodes()
# for i in [G, G2]:
#     print i.nodes()
#     print nx.is_isomorphic(i, G3, node_match=position_category)

# print nx.is_isomorphic(G1, G2)
print list(nx.bfs_edges(G, 'p1'))