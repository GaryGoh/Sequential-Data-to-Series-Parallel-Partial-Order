__author__ = 'GaryGoh'

import networkx as nx
import matplotlib.pyplot as plt
import copy
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

    def __init__(self, tree=None, name=None, num_extension=0, inclusion_probability=0.0):
        # self.data = data
        # self.left = left
        # self.right = right
        self.tree = nx.DiGraph(name)
        self.inclusion_probability = inclusion_probability
        self.heteromorphism = []

    def info(self):
        """ Return the basic info of the current tree.

        return
        -------

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
        # return str(self.left.SP_traverse()) + " " + str(self.data) + " " + str(self.right.SP_traverse())
        #
        # if self.left == None and self.right == None:
        # return str(self.data)


    def BCT_operators(self):
        """ Return a new non-isomorphic binary construction tree.

        The more details of the operation is introduced by "Global partial orders from sequential data."

        return
        -------
        BinaryConstructionTree

        """
        leaves_original = [node for node, degree in self.tree.out_degree().items() if degree == 0]

        def split(node):
            # ##
            # Return bin-tuple of the orphaned tree
            #
            # Parameter:
            # node: string
            #
            # Return:
            # (split_node_parent, node)
            # ##

            # Trivialize the orphaned subtree and re-allocate the position to
            # the child of the split_node_parent does not be cut.
            split_node_parent = M.tree.predecessors(node)[0]

            if M.tree.predecessors(split_node_parent) != []:
                split_node_parent_parent = M.tree.predecessors(split_node_parent)[0]
            else:
                split_node_parent_parent = None
            M.tree.node[node]['position'] = 'trivial'
            the_other_child = [n for n in M.tree.successors(split_node_parent) if n != node][0]

            # Shift the child to the position of its parent
            M.tree.node[the_other_child]['position'] = M.tree.node[split_node_parent]['position']
            M.tree.node[split_node_parent]['position'] = 'trivial'

            # Cutting the edge which is related to the parent of the split node
            try:
                cutting_edges_list = [(n1, n2) for n1, n2 in M.tree.edges() if
                                      (n1 == split_node_parent or n2 == split_node_parent) and (n2 != node)]

                # Remove edeges
                for i, j in cutting_edges_list:
                    M.tree.remove_edge(i, j)

                # Establish the edge on the new movement
                if split_node_parent_parent:
                    i, j = [n1 for n1, n2 in cutting_edges_list if n1 == split_node_parent_parent][0], \
                           [n2 for n1, n2 in cutting_edges_list if n2 == the_other_child][0]

                    M.tree.add_edge(i, j)

            except Exception:
                raise Exception("in the part of split() suc-function of BCT_operator() function")

            return split_node_parent, node


        def insertion(M, insert_node, split_node_parent, split_node, operator, split_node_position):
            # ##
            # Return a new binary construction tree
            #
            # Parameter:
            # split_node_parent: string, the old operator
            # split_node: string, the cut node, is used to reinsert to the binary construction tree
            # operator: string, the new operator, typically two options ("series" and "parallel")
            # split_node_position: string, the new position allocated to the cut node.
            #
            # Return:
            # M: BinaryConstructionTree
            # ##

            try:
                # Relabel the arributes of relevant nodes.
                mapping = {split_node_parent: operator}
                M.tree = nx.relabel_nodes(M.tree, mapping)
                M.tree.node[operator]['position'] = M.tree.node[insert_node]['position']
                insert_node_parent = M.tree.predecessors(insert_node)[0]

                # Re-allocate the position to the relevant nodes.
                if split_node_position == 'right' and split_node_position == M.tree.node[insert_node]['position']:
                    M.tree.node[insert_node]['position'] = 'left'

                elif split_node_position == 'left' and split_node_position == M.tree.node[insert_node]['position']:
                    M.tree.node[insert_node]['position'] = 'right'
                M.tree.node[split_node]['position'] = split_node_position

                # print M.tree.edges()
                # Remove the related edges of the current binary tree
                M.tree.remove_edge(insert_node_parent, insert_node)

                # Add the new edges to binary tree
                M.tree.add_edge(insert_node_parent, operator)
                M.tree.add_edge(operator, insert_node)

                # Eliminate the redundant edge.
                # for i, j in M.tree.edges():

            except Exception:
                raise Exception('in tha part of insertion() sub-function of BCT_operator() function')

            return M


        # To traverse all possible isomorphic binary tree
        for node in leaves_original:

            # To preserve meta structure.
            M = copy.deepcopy(self)

            # Do split operation
            split_node_parent, split_node = split(node)

            # The children that are used to be inserted
            leaves_insertion = leaves_original[:]
            leaves_insertion.remove(split_node)

            # Do insertion operation
            for insertnode in leaves_insertion:

                # Choose the operator ("series" and "parallel")
                for operator in ['series', 'parallel']:
                    operator_id = len([n for n in M.tree.nodes() if n.__contains__(operator)])
                    operator_entity = operator + str(operator_id)
                    for position in ['left', 'right']:
                        M_split = copy.deepcopy(M)
                        M_split = insertion(M_split, insertnode, split_node_parent, split_node, operator_entity,
                                            position)
                        import networkx.algorithms.isomorphism as iso

                        position_category = iso.categorical_edge_match('left', 'right')

                        if nx.is_isomorphic(self.tree, M_split.tree, edge_match=position_category):
                            # print split_node, split_node_parent, insertnode, operator_entity, position
                            self.heteromorphism.append(M_split)


        # M = copy.deepcopy(self)
        # node = 'c'
        # split_node_parent, split_node = split(node)
        # M = insertion('b', split_node_parent, split_node, "parallel2", 'right')
        # print M.tree.edges()

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
            # Regard sp_order_list as a stack.
            sp_order_stack = list(nx.dfs_postorder_nodes(self.tree, node))
            sp_order_stack.reverse()

            # New a stack to store the lower operation priority element.
            temp_stack = []

            # New a list to store the series-parallel partial order formula
            sp_order_list = []

            # New a stack to store the completed operations
            operations_stack = []

            while len(sp_order_stack) >= 2:
                # ecah round we check whether the root of the first element is match the third element.
                if len(sp_order_stack) >= 3:
                    the_first_element = sp_order_stack.pop()
                else:
                    the_first_element = temp_stack.pop()
                the_second_element = sp_order_stack.pop()
                the_third_element = sp_order_stack.pop()

                if self.tree.predecessors(the_first_element)[0] == the_third_element:
                    # if match then move these three element to sp_order_list

                    if not sp_order_list or self.tree.predecessors(sp_order_list[-1])[0] != the_first_element:
                        sp_order_list.append(the_second_element)
                        sp_order_list.append(the_first_element)
                    else:
                        sp_order_list.append(the_first_element)
                        sp_order_list.append(the_second_element)
                    sp_order_stack.append(the_third_element)

                else:
                    if self.tree.out_degree(the_first_element) > 0 and self.tree.successors(the_first_element)[
                        0] in sp_order_list and self.tree.successors(the_first_element)[1] in sp_order_list:
                        the_first_operation_element = sp_order_list.pop()
                        the_second_operation_element = sp_order_list.pop()
                        operations_stack.append(
                            [the_second_operation_element, the_first_operation_element, the_first_element])
                        temp_stack.append(the_first_element)
                    else:
                        if len(temp_stack) <= 0:
                            # if temp_stack is null then store the first element to temp_stack
                            # and put the other two back to sp_order_stack.
                            temp_stack.append(the_first_element)
                        else:
                            # get a element from temp_stack to check if match
                            the_temp_element = temp_stack.pop()
                            the_temp_operation_element = [[i, j, k] for i, j, k in operations_stack if
                                                          k == the_temp_element]

                            # To check the checked node is already a tree that was search.
                            if self.tree.predecessors(the_temp_element)[
                                0] == the_second_element and the_temp_operation_element:


                                # To check if the node shared the same parent is a tree
                                # if so, then add its children before it.
                                # otherwise add itself after the output list.
                                the_temp_operation_second_element = [[i, j, k] for i, j, k in operations_stack if
                                                                     k == the_first_element]
                                if the_temp_operation_second_element:
                                    i, j = the_temp_operation_second_element[0][0], \
                                           the_temp_operation_second_element[0][1]
                                    sp_order_list.append(i)
                                    sp_order_list.append(j)

                                sp_order_list.append(the_first_element)

                                i, j, k = the_temp_operation_element[0][0], the_temp_operation_element[0][1], \
                                          the_temp_operation_element[0][2]

                                sp_order_list.append(i)
                                sp_order_list.append(j)
                                sp_order_list.append(k)

                            # if the checked node match its parent
                            elif self.tree.predecessors(the_temp_element)[0] == the_second_element:
                                sp_order_list.append(the_first_element)
                                sp_order_list.append(the_temp_element)

                            # otherwise, put the_temp_element and the_first_element to temp_stack,
                            # as they all are not matched the parent searched so far
                            else:
                                temp_stack.append(the_temp_element)
                                temp_stack.append(the_first_element)

                    # Recover the sp_order_stack to process next search.
                    sp_order_stack.append(the_third_element)
                    sp_order_stack.append(the_second_element)

            sp_root = sp_order_stack.pop()
            sp_order_list.append(sp_root)

            return sp_order_list


        except Exception:
            raise TypeError("There is no {} in the binary construction tree".format(node))


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


    def plot_out(self, title=None, name="Output_png"):

        try:
            plt.title(str(title))
            nx.draw_graphviz(self.tree, prog='dot', with_labels=True, datas=True)
            plt.savefig(name)
            plt.show()
        except Exception:
            raise Exception("in the part of plot_out() function")


def inclusion_probability(M, M_i):
    """ Return a float value of inclusion probability.

    pi_i = n / N,
    where n = |sample i| = |M_i|
          N = |population| = |M|

    Parameters
    ------------

    M: Binary Construction Tree

    M_i: Binary Construction Tree


    return
    -------
    inclustion probability (pv_i): float

    """

    n = M_i.tree.number_of_nodes()
    N = M.tree.number_of_nodes()

    return float(n) / N


def probability_of_generating_containing_events(M_i, s):
    """ Return a float value of inclusion probability.

    f(pv, s) = product_{v in s}(pv) * product_{v not in s}(1 - pv)

    Parameters
    ------------

    M_i: Binary Construction Tree

    s: String, the incoming sequence


    return
    -------
    probability_of_generation: float

    """

    # v_in_s =
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
    root = M.get_nodes_from_position('root')[0]
    sp_order_formula = M.series_partial_order_representation(root)
    sp_order_formula.reverse()

    temp_stack = []
    # print sp_order_formula

    while len(sp_order_formula) >= 3:

        # Initialize the number of extension
        left = sp_order_formula.pop()
        if M.tree.node[left]['num_extension'] == 0:
            M.tree.node[left]['num_extension'] = 1

        right = sp_order_formula.pop()
        if M.tree.node[right]['num_extension'] == 0:
            M.tree.node[right]['num_extension'] = 1

        operator = sp_order_formula.pop()

        # When series structure
        if operator.__contains__('series') or operator.__contains__('parallel'):
            try:
                # Employ the property of series-parallel partial order to calculate the number of extension
                if operator.__contains__('series'):
                    num_extension = (M.tree.node[left]['num_extension']) * (M.tree.node[right]['num_extension'])
                    M.tree.node[operator]['num_extension'] = num_extension

                    sp_order_formula.append(operator)

                if operator.__contains__('parallel'):
                    # n1, n2 is the number of events (labels) on a partial order, we need to store the previous result.
                    n1 = len(M.series_partial_order_representation(left))
                    n2 = len(M.series_partial_order_representation(right))

                    num_extension = (math.factorial(n1 + n2) / (math.factorial(n1) * (math.factorial(n2)))) * (
                        M.tree.node[left]['num_extension']) * (M.tree.node[right]['num_extension'])

                    M.tree.node[operator]['num_extension'] = num_extension
                    sp_order_formula.append(operator)
            except Exception:
                raise Exception

        # When parallel structure
        else:
            if len(temp_stack) > 0:
                temp_left = temp_stack.pop()

                # When they share the same parent then push the temp_left to sp_order_formula
                if M.tree.predecessors(temp_left)[0] == M.tree.predecessors(left)[0]:
                    sp_order_formula.append(operator)
                    sp_order_formula.append(right)
                    sp_order_formula.append(left)
                    sp_order_formula.append(temp_left)
            # otherwise, keep pushing the new operator(parent) to temp_stack
            else:
                temp_stack.append(left)
                sp_order_formula.append(operator)
                sp_order_formula.append(right)

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
G1 = BinaryConstructionTree()

G.tree.add_node("parallel", position="root", num_extension=0)
G.tree.add_node("a", position="left", num_extension=0)
G.tree.add_node("series", position="left", num_extension=0)
G.tree.add_node("b", position="right", num_extension=0)
G.tree.add_node("c", position="right", num_extension=0)

G.tree.add_edge("parallel", "series")
G.tree.add_edge("parallel", "b")
G.tree.add_edge("series", "a")
G.tree.add_edge("series", "c")

# G1.tree.add_edge("parallel", "series")
# G1.tree.add_edge("parallel", "b")

# root = G.get_nodes_from_position("root")[0]
# sp_list = G.series_partial_order_representation(root)
# sp_list.reverse()
# print sp_list
# sp_list.insert(0, 'm')
# print sp_list

# print number_of_extensions(G)
# G1.inclusion_probability = inclusion_probability(G, G1)
# print G1.inclusion_probability
# print G.inclusion_probability
# print G.tree.edges()
# G.BCT_operators()
# print G.heteromorphism
# print len(G.heteromorphism)
# for g in G.heteromorphism:
# g.plot_out(G.heteromorphism.index(g))
# print g.tree.edges()
# print g.tree.nodes(data=True)
# print g.tree.nodes()
# root = g.get_nodes_from_position('root')[0]
# print root

# print g.series_partial_order_representation(root)
# print list(nx.dfs_postorder_nodes(g.tree))
# print number_of_extensions(g)
# print G.series_partial_order_representation()
# print G.tree.edges()
# print G.series_partial_order_representation()
# print number_of_extensions(G)

# G.heteromorphism[3].plot_out()
# print
# print G.tree.edges()

# G.plot_out("BCT_operation", "BCT_operation")

# print G.series_partial_order_representation()

# print G.tree.node['a']['number_of_extensions']

# print G.tree.successors(root)
# H = G.tree.subgraph([n for n in G.tree.nodes() if n != root])
# print H.edges()
# print root
# print G.info()


G1.tree.add_node("parallel", position="root", num_extension=0)
G1.tree.add_node("a", position="left", num_extension=0)
G1.tree.add_node("series", position="right", num_extension=0)
G1.tree.add_node("parallel1", position="left", num_extension=0)
G1.tree.add_node("series2", position="right", num_extension=0)
G1.tree.add_node("series1", position="left", num_extension=0)
G1.tree.add_node("e", position="right", num_extension=0)
G1.tree.add_node("parallel2", position="left", num_extension=0)
G1.tree.add_node("g", position="right", num_extension=0)
G1.tree.add_node("h", position="left", num_extension=0)
G1.tree.add_node("i", position="right", num_extension=0)
G1.tree.add_node("j", position="left", num_extension=0)
G1.tree.add_node("k", position="right", num_extension=0)

G1.tree.add_edge("parallel", "a")
G1.tree.add_edge("parallel", "series")
G1.tree.add_edge("series", "parallel1")
G1.tree.add_edge("series", "series2")
G1.tree.add_edge("series1", "parallel2")
G1.tree.add_edge("series1", "g")
G1.tree.add_edge("parallel1", "series1")
G1.tree.add_edge("parallel1", "e")
G1.tree.add_edge("parallel2", "h")
G1.tree.add_edge("parallel2", "i")
G1.tree.add_edge("series2", "j")
G1.tree.add_edge("series2", "k")

# print G1.series_partial_order_representation('parallel')
print number_of_extensions(G1)
# G1.plot_out()