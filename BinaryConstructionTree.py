__author__ = 'GaryGoh'

import networkx as nx
import matplotlib.pyplot as plt
import networkx.algorithms.isomorphism as iso

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


    def dfs_leaves(self, node=None):
        """ Return a list of leaves based on deep-first search algorithm.

        the order is from left to right


        parameter
        -------------
        Node: String, a node of the binary decomposition tree, default is root.

        return
        -------
        leaves: list, a list of leaves (events)

        """

        if not node:
            node = self.get_nodes_from_position('root')[0]

        bfs_all_nodes = list(nx.dfs_edges(self.tree, node))
        leaves = [j for i, j in bfs_all_nodes if self.tree.out_degree(j) == 0]

        # new a list to reduce the duplicate leaves.
        leaves_output = []
        for i in leaves:
            if i not in leaves_output:
                leaves_output.append(i)

        return leaves_output

    def dfs_operators(self, node=None):
        """ Return a list of operators based on deep-first search algorithm.

        the order is from left to right


        parameter
        -------------
        Node: String, a node of the binary decomposition tree, default is root.

        return
        -------
        operators: list, a list of operators

        """

        if not node:
            node = self.get_nodes_from_position('root')[0]

        bfs_all_nodes = list(nx.dfs_edges(self.tree, node))
        operators = [i for i, j in bfs_all_nodes]
        # operators = [i for i, j in bfs_all_nodes if self.tree.out_degree(j) == 0]


        # new a list to reduce the duplicate operators.
        operators_output = []
        for i in operators:
            if i not in operators_output:
                operators_output.append(i)

        return operators_output


    def arithmetic_expression(self, node=None):
        """ Return a list of events and operators of the arithmetic expression.

        the order is from left to right


        parameter
        -------------
        Node: String, a node of the binary decomposition tree, default is root.

        return
        -------
        arithmetic_expression: list, a list of events and operators

        """

        if not node:
            node = self.get_nodes_from_position('root')[0]

        leaves = self.dfs_leaves()
        arithmetic_expression = []

        for i in leaves:

            arithmetic_expression.append(i)

            parent = self.tree.predecessors(i)[0]
            if not parent in arithmetic_expression:
                arithmetic_expression.append(parent)

        return arithmetic_expression

    def identity_isomorphic_order(self):
        series_composition = [n for n in self.tree.nodes() if n.__contains__('series')]
        parallel_composition = [n for n in self.tree.nodes() if n.__contains__('parallel')]

        identity_series_order = self.series_partial_order_position_representation()
        for p in parallel_composition:
            for n in self.tree.successors(p):
                if n in identity_series_order and self.tree.out_degree(n) == 0:
                    identity_series_order.remove(n)
            if p in identity_series_order:
                identity_series_order.remove(p)
        for s in series_composition:
            if s in identity_series_order:
                identity_series_order.remove(s)
        return identity_series_order

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

        # To preserve meta structure.
        M_self = copy.deepcopy(self)

        # To traverse all possible isomorphic binary tree
        for node in leaves_original:

            M = copy.deepcopy(M_self)

            # Do split operation
            split_node_parent, split_node = split(node)

            # The children that are used to be inserted
            leaves_insertion = leaves_original[:]
            leaves_insertion.remove(split_node)

            # print leaves_insertion
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

                        is_heteromorphism = 0

                        identity_series_order = M_split.identity_isomorphic_order()

                        for m in self.heteromorphism:
                            m_identity_series_order = m.identity_isomorphic_order()
                            # print "m_identity:{}".format(m_identity_series_order)
                            # print "M_split:{}".format(identity_series_order)
                            if identity_series_order != m_identity_series_order:
                                is_heteromorphism += 1

                        if is_heteromorphism == len(self.heteromorphism):
                            self.heteromorphism.append(M_split)
                            # print M_split.series_partial_order_position_representation()
                            # print M_split.tree.nodes(data = True)
                            # print

        return

    def series_partial_order_position_representation(self, root=None):

        # try:
        # Regard sp_order_list as a stack.
        # sp_order_stack = list(nx.dfs_preorder_nodes(self.tree, node))
        # sp_order_stack.reverse()
        # print sp_order_stack
        if not root:
            root = self.get_nodes_from_position('root')[0]
            self.reset_nodes_visit_default()

        sp_order_stack = self.tree.successors(root)
        sp_order_stack.reverse()

        # print "current out stack:{}".format(sp_order_stack)

        sp_order_list = []
        # root = self.get_nodes_from_position('root')[0]
        # sp_order_list.append(root)
        # sp_order_stack.remove(root)

        while sp_order_stack != []:

            node = sp_order_stack.pop()
            # print "current stack:{}".format(sp_order_stack)
            # print "current node: {}".format(node)
            # print

            # if self.tree.node[node]['visit']:
            #     continue
            #
            # if self.tree.node[node]['position'] == 'left':
            #     if node.__contains__('parallel') or node.__contains__('series'):
            #         nodes = self.series_partial_order_position_representation(node)
            #         print "current nodes:{}".format(nodes)
            #         for i in nodes:
            #             sp_order_stack.append(i)
            #             self.tree.node[i]['visit'] = True
            #
            #     else:
            #         sp_order_list.append(node)
            #         self.tree.node[node]['visit'] = True
            #
            # elif self.tree.node[node]['position'] == 'right':
            #     parent = self.tree.predecessors(node)[0]
            #     left_node = self.tree.successors(parent)
            #     left_node.remove(node)
            #     left_node = left_node[0]
            #
            #     if self.tree.node[left_node]['visit']:
            #         sp_order_list.append(node)
            #         sp_order_list.append(parent)
            #     else:
            #         sp_order_stack.insert(0, node)


            if node.__contains__('parallel') or node.__contains__('series'):
                nodes = self.series_partial_order_position_representation(node)
                # print "currentnodes: {}".format(nodes)
                # if nodes[-1].__contains__('parallel'):
                #     nodes.remove(nodes[-1])
                #     sp_order_list.append(nodes)
                #     continue
                # if nodes[-1].__contains__('series'):
                #     nodes.remove(nodes[-1])
                for i in nodes:
                    sp_order_list.append(i)
                self.tree.node[node]['visit'] = True

            else:
                if self.tree.node[node]['position'] == 'left':
                    sp_order_list.append(node)
                    self.tree.node[node]['visit'] = True

                elif self.tree.node[node]['position'] == 'right':
                    #
                    parent = self.tree.predecessors(node)[0]
                    left_node = self.tree.successors(parent)
                    left_node.remove(node)
                    left_node = left_node[0]

                    if self.tree.node[left_node]['visit']:
                        sp_order_list.append(node)
                        # sp_order_list.append(parent)
                        self.tree.node[node]['visit'] = True
                        self.tree.node[parent]['visit'] = True

                    else:
                        # sp_order_list.append(left_node)
                        # sp_order_stack.remove(left_node)
                        #     sp_order_list.append(node)
                        sp_order_stack.insert(0, node)

                        # print node


        # except Exception:
        # raise Exception
        # print "Current output:{}".format(sp_order_list)
        # print
        return sp_order_list

    def reset_nodes_visit_default(self):

        for node in self.tree.nodes():
            self.tree.node[node]['visit'] = False
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
            sp_order_stack = list(nx.dfs_preorder_nodes(self.tree, node))
            sp_order_stack.reverse()

        # # New a stack to store the lower operation priority element.
        # temp_stack = []
        #
        # # New a list to store the series-parallel partial order formula
        # sp_order_list = []
        #
        # # New a stack to store the completed operations
        # operations_stack = []
        #
        # while len(sp_order_stack) + len(temp_stack) >= 2:
        # # ecah round we check whether the root of the first element is match the third element.
        # # if len(sp_order_stack) >= 3:
        # #     the_first_element = sp_order_stack.pop()
        # # else:
        # #     the_first_element = temp_stack.pop()
        # # the_second_element = sp_order_stack.pop()
        # # the_third_element = sp_order_stack.pop()
        #
        #     current_elements = []
        #
        #     while len(current_elements) < 3:
        #         if sp_order_stack:
        #             current_elements.append(sp_order_stack.pop())
        #         else:
        #             current_elements.append(temp_stack.pop())
        #     current_elements.reverse()
        #
        #     print current_elements
        #
        #     print "Current: {} {} {} {}".format(current_elements[0], current_elements[1], current_elements[2],
        #                                         sp_order_stack)
        #     # if match then move the first two elements to sp_order_list
        #     if self.tree.predecessors(current_elements[0])[0] == current_elements[2]:
        #
        #         if not sp_order_list or self.tree.predecessors(sp_order_list[-1])[0] != current_elements[0]:
        #             sp_order_list.append(current_elements[1])
        #             # if self.tree.out_degree(the_first_element) > 0 and len(sp_order_stack) == 0:
        #             #     the_temp_operation_final_element = [[i, j, k] for i, j, k in operations_stack if
        #             #                                         k == the_first_element]
        #             #     i, j = the_temp_operation_final_element[0][0], the_temp_operation_final_element[0][1]
        #             #
        #             #     sp_order_list.append(i)
        #             #     sp_order_list.append(j)
        #             sp_order_list.append(current_elements[0])
        #
        #         else:
        #             if self.tree.out_degree(current_elements[0]) > 0 and len(sp_order_stack) == 0:
        #                 the_temp_operation_final_element = [[i, j, k] for i, j, k in operations_stack if
        #                                                     k == current_elements[0]]
        #                 i, j = the_temp_operation_final_element[0][0], the_temp_operation_final_element[0][1]
        #
        #                 sp_order_list.append(i)
        #                 sp_order_list.append(j)
        #
        #             sp_order_list.append(current_elements[0])
        #             sp_order_list.append(current_elements[1])
        #         sp_order_stack.append(current_elements[2])
        #
        #     else:
        #         if self.tree.out_degree(current_elements[0]) > 0 and self.tree.successors(current_elements[0])[
        #             0] in sp_order_list and self.tree.successors(current_elements[0])[1] in sp_order_list:
        #             # if the_first_element is the sub-tree that was already searched than
        #             # push to operations_stack and put the operator as a stamp to Temp_stack.
        #
        #             # else:
        #             the_first_operation_element = sp_order_list.pop()
        #             the_second_operation_element = sp_order_list.pop()
        #             operations_stack.append(
        #                 [the_second_operation_element, the_first_operation_element, current_elements[0]])
        #             temp_stack.append(current_elements[0])
        #
        #             stamps_in_operations_stack = [k for i, j, k in operations_stack]
        #             if stamps_in_operations_stack and current_elements[0] in stamps_in_operations_stack:
        #
        #                 # if stamps_in_operations_stack is not null then temp_stack can not be null
        #                 the_recover_element = temp_stack.pop()
        #
        #                 # in case of duplicate of the_first_element.
        #                 if temp_stack and the_recover_element == current_elements[0]:
        #                     the_recover_element = temp_stack.pop()
        #
        #             # Recover the sp_order_stack to process next search.
        #             sp_order_stack.append(current_elements[2])
        #             sp_order_stack.append(current_elements[1])
        #             if the_recover_element:
        #                 sp_order_stack.append(current_elements[0])
        #                 sp_order_stack.append(the_recover_element)
        #             sp_order_stack.append(current_elements[0])
        #
        #         else:
        #             if len(temp_stack) <= 0:
        #                 # if temp_stack is null then store the first element to temp_stack
        #                 # and put the other two back to sp_order_stack.
        #
        #                 temp_stack.append(current_elements[0])
        #             else:
        #                 # get a element from temp_stack to check if match
        #                 # No need to check if sp_order_list is null as at the final round,
        #                 # they must share the same parent(root).
        #
        #                 the_temp_element = temp_stack.pop()
        #                 the_temp_operation_element = [[i, j, k] for i, j, k in operations_stack if
        #                                               k == the_temp_element]
        #
        #                 # To check the checked node is already a tree that was search.
        #                 if self.tree.predecessors(the_temp_element)[
        #                     0] == current_elements[1] and the_temp_operation_element:
        #
        #
        #                     # To check if the node shared the same parent is a tree
        #                     # if so, then add its children before it.
        #                     # otherwise add itself after the output list.
        #                     the_temp_operation_second_element = [[i, j, k] for i, j, k in operations_stack if
        #                                                          k == current_elements[0]]
        #
        #                     if the_temp_operation_second_element:
        #                         i, j = the_temp_operation_second_element[0][0], \
        #                                the_temp_operation_second_element[0][1]
        #                         sp_order_list.append(i)
        #                         sp_order_list.append(j)
        #                         operations_stack.remove(the_temp_operation_second_element[0])
        #                     sp_order_list.append(current_elements[0])
        #
        #                     i, j, k = the_temp_operation_element[0][0], the_temp_operation_element[0][1], \
        #                               the_temp_operation_element[0][2]
        #
        #                     sp_order_list.append(i)
        #                     sp_order_list.append(j)
        #                     sp_order_list.append(k)
        #                     operations_stack.remove(the_temp_operation_element[0])
        #
        #                 # if the checked node match its parent
        #                 elif self.tree.predecessors(the_temp_element)[0] == current_elements[1]:
        #                     sp_order_list.append(current_elements[0])
        #                     sp_order_list.append(the_temp_element)
        #
        #                 # otherwise, put the_temp_element and the_first_element to temp_stack,
        #                 # as they all are not matched the parent searched so far
        #                 else:
        #                     temp_stack.append(the_temp_element)
        #                     temp_stack.append(current_elements[0])
        #
        #             # Recover the sp_order_stack to process next search.
        #             sp_order_stack.append(current_elements[2])
        #             sp_order_stack.append(current_elements[1])
        #
        #
        #     print "Temp_stack: {}".format(temp_stack)
        #     print "operations_stack: {}".format(operations_stack)
        #     print "Output: {}".format(sp_order_list)
        #     print
        #
        #     # current_elements.remove(current_elements[0])
        #     # current_elements.remove(current_elements[1])
        #     # current_elements.remove(current_elements[2])
        #
        # sp_root = sp_order_stack.pop()
        # sp_order_list.append(sp_root)
        #
        # while operations_stack:
        #     i, j, k = operations_stack.pop()
        #     sp_order_list.insert(sp_order_list.index(k), i)
        #     sp_order_list.insert(sp_order_list.index(k), j)

        except Exception:
            raise TypeError("There is no {} in the binary construction tree".format(node))
        return sp_order_stack


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


def inclusion_probability(M, s):
    """ Return a float value of inclusion probability.

    pi_i = n / N,
    where n = |sequence| = |s|
          N = |population| = |M|

    Parameters
    ------------

    M: Binary Construction Tree

    s: String, A sequence without inclusion probability


    return
    -------
    A dictionary of the sequence with (event: inclusion probability)

    """

    # initialize a dictionary to store events with inclusion probabilites
    s_with_inclusion_probabilites = {}

    # calculating the events of intersection and difference.
    V = [i for i in M.tree.nodes() if not (i.__contains__('parallel') or i.__contains__('series'))]
    s_in_M = list(set(s).intersection(V))

    # the size of two samples.
    n = len(s_in_M)
    N = len(V)

    pv = float(n) / N

    for i in s:
        if i in s_in_M:
            s_with_inclusion_probabilites[i] = pv
        else:
            # in fact that, the probabilities should be 0,
            # however, for convenience, we define the probability as 1 - pv
            s_with_inclusion_probabilites[i] = 1 - pv

    return s_with_inclusion_probabilites


def probability_of_generating_containing_events(M, s):
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

    # initialize the probabilities of generating containing events.
    f = 1

    s_with_inclusion_probabilities = inclusion_probability(M, s)
    for v, p in s_with_inclusion_probabilities.items():
        f *= p

    return f


def compatable_with_SP(M, s):
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

    # To ensure partial order M has heteromorphic
    # if not M.heteromorphism:
    # M.BCT_operators()

    # for m in M.heteromorphism:

    m_leaves = M.dfs_leaves()
    print m_leaves
    print s[0], s[-1]

    if s[0] in m_leaves:
        m_leaves = m_leaves[m_leaves.index(s[0]):]
    else:
        return 0

    # Traversal the incoming sequence, regarded it as a total order of a event set.
    index = 0
    for i in m_leaves:
        print i
        if i == s[index]:
            continue
        else:
            parent = M.tree.predecessors(i)[0]

            if parent.__contains__('parallel'):
                children = M.tree.successors(parent)
                if s[index] in children:
                    # for child in children:
                    # m_leaves.remove(child)
                    continue
                else:
                    return 0
        index += 1
        return 1


def number_of_extensions(M, root=None):
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

    sp_order_formula = [i for i in M.series_partial_order_representation(root) if
                        (i.__contains__('series') or i.__contains__('parallel'))]

    while sp_order_formula:

        # Extend the children of the current operator
        operator = sp_order_formula.pop()
        left, right = M.tree.successors(operator)

        for child in [left, right]:

            # Recursion if the child is a operator (also known as a sub-tree in Binary Construction Tree).
            if M.tree.node[child]['num_extension'] == 0:
                if child.__contains__('series') or child.__contains__('parallel'):
                    M.tree.node[child]['num_extension'] = number_of_extensions(M, child)
                else:
                    M.tree.node[child]['num_extension'] = 1

            # Employ the property of series-parallel partial order to calculate the number of extension
            if operator.__contains__('series'):
                num_extension = (M.tree.node[left]['num_extension']) * (M.tree.node[right]['num_extension'])
                M.tree.node[operator]['num_extension'] = num_extension

            if operator.__contains__('parallel'):
                # n1, n2 is the number of events (labels) on a partial order, we need to store the previous result.
                n1 = len([i for i in M.series_partial_order_representation(left) if
                          not (i.__contains__('series') or i.__contains__('parallel'))])
                n2 = len([i for i in M.series_partial_order_representation(right) if
                          not (i.__contains__('series') or i.__contains__('parallel'))])

                num_extension = (math.factorial(n1 + n2) / (math.factorial(n1) * (math.factorial(n2)))) * (
                    M.tree.node[left]['num_extension']) * (M.tree.node[right]['num_extension'])

                M.tree.node[operator]['num_extension'] = num_extension
    #
    # # When series structure
    # if operator.__contains__('series') or operator.__contains__('parallel'):
    # try:
    # # Employ the property of series-parallel partial order to calculate the number of extension
    # if operator.__contains__('series'):
    # num_extension = (M.tree.node[left]['num_extension']) * (M.tree.node[right]['num_extension'])
    # M.tree.node[operator]['num_extension'] = num_extension
    #
    #             if operator.__contains__('parallel'):
    #                 # n1, n2 is the number of events (labels) on a partial order, we need to store the previous result.
    #                 n1 = len(M.series_partial_order_representation(left))
    #                 n2 = len(M.series_partial_order_representation(right))
    #
    #                 num_extension = (math.factorial(n1 + n2) / (math.factorial(n1) * (math.factorial(n2)))) * (
    #                     M.tree.node[left]['num_extension']) * (M.tree.node[right]['num_extension'])
    #
    #                 M.tree.node[operator]['num_extension'] = num_extension
    #
    #             print "{}:{}".format(operator, num_extension)
    #             sp_order_formula.append(operator)
    #         except Exception:
    #             raise Exception
    #
    #     # When parallel structure
    #     else:
    #         if len(temp_stack) > 0:
    #             temp_left = temp_stack.pop()
    #
    #             # When they share the same parent then push the temp_left to sp_order_formula
    #             if M.tree.predecessors(temp_left)[0] == M.tree.predecessors(left)[0]:
    #                 sp_order_formula.append(operator)
    #                 sp_order_formula.append(right)
    #                 sp_order_formula.append(left)
    #                 sp_order_formula.append(temp_left)
    #         # otherwise, keep pushing the new operator(parent) to temp_stack
    #         else:
    #             temp_stack.append(left)
    #             sp_order_formula.append(operator)
    #             sp_order_formula.append(right)
    #
    # result = sp_order_formula.pop()
    return M.tree.node[root]['num_extension']


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
G2 = BinaryConstructionTree()

G.tree.add_node("parallel", position="root", num_extension=0, visit=False)
G.tree.add_node("a", position="right", num_extension=0, visit=False)
G.tree.add_node("series", position="left", num_extension=0, visit=False)
G.tree.add_node("b", position="right", num_extension=0, visit=False)
G.tree.add_node("c", position="left", num_extension=0, visit=False)

G.tree.add_edge("parallel", "series")
G.tree.add_edge("parallel", "b")
G.tree.add_edge("series", "a")
G.tree.add_edge("series", "c")

G2.tree.add_node("series", position="root", num_extension=0,visit=False)
G2.tree.add_node("a", position="left", num_extension=0,visit=False)
G2.tree.add_node("series1", position="left", num_extension=0,visit=False)
G2.tree.add_node("b", position="left", num_extension=0,visit=False)
G2.tree.add_node("series2", position="right", num_extension=0,visit=False)
G2.tree.add_node("c", position="right", num_extension=0,visit=False)
G2.tree.add_node("d", position="right", num_extension=0,visit=False)

G2.tree.add_edge("series", "series1")
G2.tree.add_edge("series", "d")
G2.tree.add_edge("series1", "a")
G2.tree.add_edge("series1", "series2")
G2.tree.add_edge("series2", "b")
G2.tree.add_edge("series2", "c")


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


G1.tree.add_node("parallel", position="root", num_extension=0, visit=False)
G1.tree.add_node("a", position="left", num_extension=0,visit=False)
G1.tree.add_node("series", position="right", num_extension=0,visit=False)
G1.tree.add_node("parallel1", position="left", num_extension=0,visit=False)
G1.tree.add_node("series2", position="right", num_extension=0,visit=False)
G1.tree.add_node("series1", position="left", num_extension=0,visit=False)
G1.tree.add_node("e", position="right", num_extension=0,visit=False)
G1.tree.add_node("parallel2", position="left", num_extension=0,visit=False)
G1.tree.add_node("g", position="right", num_extension=0,visit=False)
G1.tree.add_node("h", position="left", num_extension=0,visit=False)
G1.tree.add_node("i", position="right", num_extension=0,visit=False)
G1.tree.add_node("j", position="left", num_extension=0,visit=False)
G1.tree.add_node("parallel3", position="right", num_extension=0,visit=False)
G1.tree.add_node("l", position="left", num_extension=0,visit=False)
G1.tree.add_node("m", position="right", num_extension=0,visit=False)

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
G1.tree.add_edge("series2", "parallel3")
G1.tree.add_edge("parallel3", "l")
G1.tree.add_edge("parallel3", "m")


#
# G1.tree.add_node("parallel", position="root", num_extension=0)
# G1.tree.add_node("a", position="left", num_extension=0)
# G1.tree.add_node("series", position="right", num_extension=0)
# G1.tree.add_node("parallel1", position="left", num_extension=0)
# G1.tree.add_node("series2", position="right", num_extension=0)
# G1.tree.add_node("series1", position="left", num_extension=0)
# G1.tree.add_node("e", position="right", num_extension=0)
# G1.tree.add_node("parallel2", position="left", num_extension=0)
# G1.tree.add_node("g", position="right", num_extension=0)
# G1.tree.add_node("h", position="left", num_extension=0)
# G1.tree.add_node("i", position="right", num_extension=0)
# G1.tree.add_node("j", position="left", num_extension=0)
# G1.tree.add_node("k", position="right", num_extension=0)
#
# G1.tree.add_edge("parallel", "a")
# G1.tree.add_edge("parallel", "series")
# G1.tree.add_edge("series", "parallel1")
# G1.tree.add_edge("series", "series2")
# G1.tree.add_edge("series1", "parallel2")
# G1.tree.add_edge("series1", "g")
# G1.tree.add_edge("parallel1", "series1")
# G1.tree.add_edge("parallel1", "e")
# G1.tree.add_edge("parallel2", "h")
# G1.tree.add_edge("parallel2", "i")
# G1.tree.add_edge("series2", "j")
# G1.tree.add_edge("series2", "k")

# print G1.series_partial_order_representation('parallel')
# print list(nx.dfs_postorder_nodes(G1.tree, 'series'))
# print
# print G1.series_partial_order_representation('series')
# root = G1.get_nodes_from_position('root')[0]
# print number_of_extensions(G1, root)
# s = ['a', 'b', 'c']
s = ['c', 'a']


# s = ['b', 'a', 'c', 'd']

# print G.tree.nodes()
# print inclusion_probability(G,s)
# print probability_of_generating_containing_events(G, s)
# G1.BCT_operators()
# print G1.series_partial_order_representation()

# print G1.tree.edges()

# print G1.series_partial_order_position_representation()


G.BCT_operators()
print len(G.heteromorphism)

# print G2.dfs_leaves('series2')
# print G.identity_isomorphic_order()

# print G2.series_partial_order_position_representation()
# print G.identity_isomorphic_order()
# print G2.series_partial_order_position_representation()
# print G.identity_isomorphic_order()

# print G2.series_partial_order_position_representation()
# print G.identity_isomorphic_order()
# print G.plot_out()

# print compatable_with_SP(G, s)

for g in G.heteromorphism:
#     print g.tree.nodes(data=True)
    print g.series_partial_order_position_representation()
#     print g.tree.edges()
#     print

# print
# print len(g.heteromorphism)

# print g.series_partial_order_position_representation()

# print


# print g.tree.nodes()

# G1.plot_out()