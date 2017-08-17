"""
Distribution: Set of all possible random variables
Example:
    Flipping Coins for heads and tails:
    a binomial distribution(two possible outcomes)  # 二项分布
    discrete (categories of heads and tails, no real numbers) # 离散
    evenly weighted (heads are just as likely as tails)  # 权重是相等的
    Tornado events in Ann Arbor # 安阿伯(美国密歇根州东南部一城市)
    a binomial distribution
    Discrete
    evenly weighted(tornadoes are rare events)
"""
import pandas as pd
import numpy as np


np.random.binomial(1, 0.5)
# The first is the number of times we want it to run
# The second is the chance we get a zero, which we will use to represent heads here
# The third
# 返回值是成功的次数
# np.random.binomial(n, p, size)
"""
	
np.random.binomial(N, p, size = q)
np.random.binomial(1, p, size = q)
np.random.binomial(N,p, size= q)
1st and 3rd are similar, i can see. These two are binomial random number generator

And, 2nd one is bernoulli random number generator

Explanation of binomial:

A binomial random variable counts how often a particular event occurs in a fixed number of tries or trials.

Here,

n = number of trials
p = probability event of interest occurs on any one trial
size = number of times you want to run this experiment
Suppose, You wanna check how many times you will get six if you roll dice 10 times. Here,

n = 10,
p = (1/6) # probability of getting six in each roll
But, You have to do this experiment multiple times.

Let, In 1st experiment, you get 3 six

In 2nd expwriment, you get 2 six

In 3rd experiment, you get 2 six

In Pth experiment, you get 2 six, here P is the size
"""
chance_of_tornado = 0.01
tornado_events = np.random.binomial(1, chance_of_tornado, 1000000)

two_days_in_a_row = 0
for j in range(1, len(tornado_events) - 1):
    if tornado_events[j]==1 and tornado_events[j-1]==1:
        two_days_in_a_row += 1
print("{} tornadoes back to back in {} years.".format(two_days_in_a_row, 1000000/365))
