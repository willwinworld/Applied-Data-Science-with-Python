import pandas as pd
import numpy as np
import re
"""
where the x axis is the value of the observation
and y axis represents the probability that a
given observation will occur
if all numbers are equally likely to be drawn
when you sample from it.
this should be graphed as a flat horizontal line
and this flat line is actually called the uniform distribution
take the normal distribution which is also called the
Gaussian Distribution or Bell curve
"""


# with open("university_towns.txt", "r") as f:
#     data = []
#     for line in f:
#         m = re.search('(.+)\[edit\]', line)
#         if m:
#             state = m.group(1)
#         else:
#             town = line.split('(')[0].strip()  # 当没有匹配到state时，上一个在解释器中的state还并没有消失
#             print(data)
#             data.append([state, town])
#     print(data)


def copy(origin):
    """
    实现复制字符串
    """
    origin_list = [x for x in origin]
    new = "".join(origin_list)
    return new


def flat(origin):
    """
    [[['a'],['b'], ['c']]]  --> ['a', 'b', 'c']
    """
    if len(origin) != 3:
        return flat(origin[0])
        # 递归调用
    else:
        res = []
        for i in origin:
            res.append(i[0])
        return res


def search(a, n, key):
    a.insert(0, key)  # 哨兵
    print(a)
    i = n  # 列表长度
    while a[i] != key:
        i -= 1
    # 返回0就是查找失败
    return i


if __name__ == '__main__':
    # test = copy("a cb d")
    # print(test)
    # test = flat([[['a'], ['b'], ['c']]])
    # print(test)
    test = search([2, 3, 4], 3, 1)
    print(test)
