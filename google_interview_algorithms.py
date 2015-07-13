__author__ = 'GaryGoh'
from operator import itemgetter
from collections import deque


s = [2, 3, 7, 4, 1, 8, 0, 9, 5, 6, 3]
# print s


def bubble_sort(s):
    for i in range(0, len(s)):
        for j in range(0, len(s) - (i + 1)):
            if s[j] > s[j + 1]:
                s[j] ^= s[j + 1]
                s[j + 1] ^= s[j]
                s[j] ^= s[j + 1]
    return s


def selection_sort(s):
    # for i in range(len(s) -1):
    # min_index = i
    #     for j in range(i+1, len(s)):
    #         if s[min_index] > s[j]:
    #             min_index = j
    #     if min_index != i:
    #         s[min_index], s[i] = s[i], s[min_index]

    for i in range(len(s) - 1):
        min_index = min(enumerate(s[i + 1: len(s)]), key=itemgetter(1))[0] + i + 1
        if min_index != i:
            s[min_index], s[i] = s[i], s[min_index]

    return s


def quick_sort2(s, left, right):
    pivot = left

    while i != j:
        for i in range(right, pivot, -1):
            if s[i] < s[pivot]:
                i, pivot = pivot, i

        for j in range(left, pivot):
            if s[j] > s[pivot]:
                j, pivot = pivot, j


def quick_sort(s):
    if len(s) <= 1:
        return s
    else:
        pivot = s[0]
        return quick_sort([i for i in s[1:] if i < pivot]) + [pivot] + quick_sort([i for i in s[1:] if i >= pivot])

# print bubble_sort(s)
# print selection_sort(s)
# print quick_sort(s)


graph = {
        '1': ['2', '3', '4'],
        '2': ['5', '6'],
        '5': ['9', '10'],
        '4': ['7', '8'],
        '7': ['11', '12']
        }



def bfs(graph, start, end):
    queue = []

    queue.append([start])

    while queue:
        path = queue.pop(0)
        # print path
        current_node = path[-1]
        # print current_node
        if current_node == end:
            return path
        else:
            for adj_node in graph.get(current_node, []):
                inqueue_path = list(path)
                inqueue_path.append(adj_node)
                queue.append(inqueue_path)

def dfs(graph, start, end):
    queue = []

    queue.append([start])

    while queue:
        path = queue.pop()
        # print path
        current_node = path[-1]
        # print current_node
        if current_node == end:
            return path
        else:
            for adj_node in graph.get(current_node, []):
                inqueue_path = list(path)
                inqueue_path.append(adj_node)
                queue.append(inqueue_path)


# print bfs(graph, '1', '12')
# print dfs(graph, '1', '12')

def fib(n):
    if n <= 2:
        return 1
    else:
        return fib(n - 1) + fib(n-2)

print fib(4)