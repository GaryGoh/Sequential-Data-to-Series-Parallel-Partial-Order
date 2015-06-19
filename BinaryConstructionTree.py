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

    data: string, optional
        the value of current node.

    left: string, optional
        the left child of the current node.

    right: string, optional
        the right child of the current node.

    """

    def __init__(self, data=None, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right


    def __str__(self):
        return str(self.data)


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


t = BinaryConstructionTree("p", BinaryConstructionTree('s', BinaryConstructionTree(2), BinaryConstructionTree(5)),
                           BinaryConstructionTree(3))

print t.SP_traverse()