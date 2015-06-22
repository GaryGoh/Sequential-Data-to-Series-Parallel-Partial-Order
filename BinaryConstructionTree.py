__author__ = 'GaryGoh'


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

    def __init__(self, data=None, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right

    def info(self):
        """ Return the basic info of the current tree.

        return
        -------
        data: the value of current node.

        left: the left child of the current node.

        right: the right child of the current node.

        """
        return self.data, self.left.data, self.right.data


    def children(self):
        """ Return the children of the current node.

        """
        return self.left, self.right

    def __iter__(self):
        """ Create an iterator of the tree(from the left child to the right child).

        return
        -------
        The tuple combined of the children
        """

        if not self.left and not self.right:
            raise StopIteration
        return self.children().__iter__()

    def SP_traverse(self):
        """ Return a string of series-parallel partial order.

        A recursion way to implement in-order traversal.

        return
        -------
        A simple formula of series-parallel partial order

        """
        if self.left != None and self.right == None:
            return str(self.left.SP_traverse()) + " " + str(self.data)

        if self.right != None and self.left == None:
            return str(self.data) + " " + str(self.right.SP_traverse())

        if self.left != None and self.right != None:
            return str(self.left.SP_traverse()) + " " + str(self.data) + " " + str(self.right.SP_traverse())

        if self.left == None and self.right == None:
            return str(self.data)


"""""""""""""""
Testing
"""""""""""""""
t = BinaryConstructionTree("p", BinaryConstructionTree('s', BinaryConstructionTree(2), BinaryConstructionTree(5)),
                           BinaryConstructionTree(3))

print t.info()

for i in t:
    print i.data