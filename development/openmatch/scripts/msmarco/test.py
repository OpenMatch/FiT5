import random
a = {'name': 'jinxin', 'age': 16, 'male': '男', 'high': 185, 'weight': None, 'address': '北京'}
i = 0
b = a.keys()
for k in b:
    i += 1
    del a[k]
    if i> 2: break
print(a)