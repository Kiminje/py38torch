#   n만큼 늘려주는 함수를 만들어주는 함수
def make_incrementor(n):
    return lambda x: x+n


f42 = make_incrementor(42)
print(f42(14))

tel = {'jack': 4093, 'sape': 41233, 'guido': 42318}

for (k, v) in tel.items():
    print(k, v)


for k in list(tel):
    print(k, tel[k])

# import IT_system_module as IT
# IT.fib(100)
# Result = IT.fib2(100)
from IT_SYSTEM.IT_system_module import *
fib(100)
Result = fib2(100)
print(Result)

import numpy as np

print(np.arange(10))
x = np.linspace(-3, 3, 7)
y = np.linspace(-5, 5, 7)
import matplotlib.pyplot as plt
Scat = plt.scatter(x,y)
print(Scat)