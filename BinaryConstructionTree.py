__author__ = 'GaryGoh'

import networkx as nx
import matplotlib.pyplot as plt
import math


class BinaryConstructionTree(object):
    """ Binary Construction Tree

    Using Binary Construction Tree to represent series-parallel partial order.

    Based on the benefit of Binary Construction Tree, we can recursively compute alpha(M)
    in time linear in the number of elements in the partial order.

    alpha(M) is the number of complete extension of M.

    Reference:
        Mannila, Heikki, and Christopher Meek. "Global partial orders from sequential data."
        Proceedings of the sixth ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2000.


    Parameters
    ------------

    data: var, optional
        the value of current node.

    left: BinaryConstructTree, optional
        the left child of the current node.

    right: BinaryConstructTree, optional
        the right child of the current node.

    """

    def __init__(self, tree=None, name=None, num_extension=0):
        # self.data = data
        # self.left = left
        # self.right = right
        self.tree = nx.DiGraph(name)
        self.num_extension = num_extension

    def info(self):
        """ Return the basic info of the current tree.

        return
        -------
        data: the value of current node.

        left: the left child of the current node.

        right: the right child of the current node.

        """
        return nx.info(self.tree)


        # def children(self):
        """ Return the children of the current node.

        """
        # return self.left, self.right

    def __iter__(self):
        """ Create an iterator of the tree(from the left child to the right child).

        return
        -------
        The tuple combined of the children
        """

        if not self.left and not self.right:
            raise StopIteration
        return self.children().__iter__()

        # def SP_traverse(self):
        """ Return a string of series-parallel partial order.

        A recursion way to implement in-order traversal.

        return
        -------
        A simple formula of series-parallel partial order

        """
        # if self.left != None and self.right == None:
        # return str(self.left.SP_traverse()) + " " + str(self.data)
        #
        # if self.right != None and self.left == None:
        # return str(self.data) + " " + str(self.right.SP_traverse())
        #
        # if self.left != None and self.right != None:
        #     return str(self.left.SP_traverse()) + " " + str(self.data) + " " + str(self.right.SP_traverse())
        #
        # if self.left == None and self.right == None:
        #     return str(self.data)


    def BCT_operators(nodes):
        """ Return a new non-isomorphic binary construction tree.

        The more details of the operation is introduced by "Global partial orders from sequential data."

        return
        -------
        BinaryConstructionTree

        """

        return

    def series_partial_order_representation(self, node=None):
        """ Return a path of the list of nodes.

        parameter
        -------------
        BCT_tree: BinaryConstructionTree
        position: String, optional

        return
        -------
        A list of nodes. i.e ['p1', 'a', 'b']

        """
        try:
            return list(nx.dfs_postorder_nodes(self.tree, node))
        except Exception:
            raise TypeError("There is no " + node + " in the binary construction tree")


    def get_nodes_from_position(self, position=None):
        """ Return a list of nodes of the position.

        parameter
        -------------
        position: String, optional

        return
        -------
        A list of nodes. i.e ['p1', 'a', 'b']

        """
        return [nodes for nodes, positions in self.tree.nodes(data=True) if positions["position"] == position]


def inclusion_probability(N, s):
    """ Return a float value of inclusion probability.

    pi_i = n / N,
    where n = |sample i|
          N = |population|

    Parameters
    ------------

    N: array of data (typically is string)


    return
    -------
    inclustion probability (pv_i): float

    """

    return


def compatable_with_SP(s, M):
    """ Return a int value of whether the sequence is compatible with the model M.

    Parameters
    ------------

    s: array of data (typically is string)

    M: partial order M


    return
    -------
    1: is compatible with M

    0: otherwise


    """

    return


def number_of_extensions(M):
    """ Return a int value of the number of completed extension of the SP-order.

    We store the number of extension of each sub-tree(sub-partial order) such each node can have its partial order
    information with its current number of extension.

    Parameters
    ------------

    M: partial order M


    return
    -------
    n: int


    """

    sp_order_formula = M.series_partial_order_representation()
    sp_order_formula.reverse()
    while len(sp_order_formula) >= 3:

        # Initialize the number of extension
        left = sp_order_formula.pop()
        if M.tree.node[left]['num_extension'] == 0:
            M.tree.node[left]['num_extension'] = 1

        right = sp_order_formula.pop()
        if M.tree.node[right]['num_extension'] == 0:
            M.tree.node[right]['num_extension'] = 1

        operator = sp_order_formula.pop()

        try:
            # Employ the property of series-parallel partial order to calculate the number of extension
            if operator == 'series':
                num_extension = (M.tree.node[left]['num_extension']) * (M.tree.node[right]['num_extension'])
                M.tree.node[operator]['num_extension'] = num_extension
                sp_order_formula.append(operator)
            if operator == 'parallel':
                # n1, n2 is the number of events (labels) on a partial order, we need to store the previous result.
                n1 = len(M.series_partial_order_representation(left))
                n2 = len(M.series_partial_order_representation(right))

                num_extension = (math.factorial(n1 + n2) / (math.factorial(n1) * (math.factorial(n2)))) * (
                M.tree.node[left]['num_extension']) * (M.tree.node[right]['num_extension'])

                M.tree.node[operator]['num_extension'] = num_extension
                sp_order_formula.append(operator)

        except Exception:
            raise Exception

    result = sp_order_formula.pop()

    return M.tree.node[result]['num_extension']


"""""""""""""""
Testing
"""""""""""""""
# t = BinaryConstructionTree("p", BinaryConstructionTree('s', BinaryConstructionTree(2), BinaryConstructionTree(5)),
# BinaryConstructionTree(3))

# print t.info()
#
# for i in t:
# print i.data

G = BinaryConstructionTree()

G.tree.add_node("parallel", position="root", num_extension=0)
G.tree.add_node("a", position="left", num_extension=0)
G.tree.add_node("series", position="left", num_extension=0)
G.tree.add_node("b", position="right", num_extension=0)
G.tree.add_node("c", position="right", num_extension=0)

G.tree.add_edge("parallel", "series")
G.tree.add_edge("parallel", "b")
G.tree.add_edge("series", "a")
G.tree.add_edge("series", "c")

# root = G.get_nodes_from_position("root")[0]
# sp_list = G.series_partial_order_representation(root)
# sp_list.reverse()
# print sp_list
# sp_list.insert(0, 'm')
# print sp_list
print number_of_extensions(G)

# print G.series_partial_order_representation()

# print G.tree.node['a']['number_of_extensions']

# print G.tree.successors(root)
# H = G.tree.subgraph([n for n in G.tree.nodes() if n != root])
# print H.edges()
# print root
# print G.info()