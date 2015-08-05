__author__ = 'GaryGoh'

# import networkx as nx
# from networkx.algorithms import bipartite
# import matplotlib.pyplot as plt
import math
#
#
# B = nx.Graph()
# B.add_nodes_from([1,2,3,4], bipartite=0) # Add the node attribute "bipartite"
# B.add_nodes_from(['a','b','c'], bipartite=1)
# B.add_edges_from([(1,'a'), (1,'b'), (2,'b'), (2,'c'), (3,'c'), (4,'a')])
# for i in B:
# print i
#
#
# # # same layout using matplotlib with no labels
# plt.title("draw_networkx")
# pos = nx.graphviz_layout(B, prog='dot')
# nx.draw(B, pos, with_labels=True, arrows=False)
# # nx.draw(G1,pos,with_labels=True,arrows=False)
#
# # plt.savefig('nx_test.png')
# plt.show()

a = [2, 3, 8]
a1 = [2, 8]

# print a in a1
b = [3, 4, 5, 6, 7]
c = [[1, 2, 3], [2, 3, 4]]
d = [[i, j, k] for i, j, k in c if k == 3]
f = 12

i, j, k = c[0]
a2 = "abc"

# i = a.index(a[0])
# j = a.index(a[-1])
# print a[i:]



# a.append([11,12])


# print 11 in a
# print next(a)
a2.split()
aa2 = []
for a in a2:
    aa2.append(a)

print aa2
# print a2[0]
aa = aa2.pop()
print aa
# print i, j,k
#
# if d:
# print 'yes'
# else:
#     print 'no'
# c = a[:]
# s = ['a1', 'a2', 'a3', 'b']
# print [n for n in s if n.__contains__('a')]

# a.insert(a.index(3), 6)
# a.insert(a.index(3), 7)

e = [1]

# a.append(f)
# a.append(f)
# if a:
#     print "yew"
# else:
#     print 'no'

# if len(e) > 0 and e.pop():
#     print "yes"

# c.remove(5)
# print c
# print a

# print 3 in c[0]

# print list(set(a).intersection(b))
# print list(set(a).difference(b))

# print

# s = "series"
# n = 1

# l = s + str(n)
# print l

# a = math.factorial(10)/(math.factorial(7) * math.factorial(3)) * 12
#
# print a * 12