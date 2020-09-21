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
