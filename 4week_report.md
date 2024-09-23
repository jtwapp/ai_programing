```python
# 과제1
result = np.array([])
for i in range(3):
  a = np.arange(1, 46)
  np.random.shuffle(a)
  lotto = a[:6]
  result = np.append(result, lotto)
print(result)
result = np.reshape(result, (3, 6))
print(result)
```
# 결과
<p align="left">
 <img src = "4week_report.1.jpg">
</p>

```python
# 과제2
import numpy as np
shape = (3, 3, 6)
arr = np.random.randint(0, 21, shape)
print(arr)

shape = (3, 6)
arr = np.random.randint(0, 101, shape)
print(arr)
```
# 결과
<p align="left">
 <img src = "4week_report.2.jpg">
</p>

```python
# 과제3
import numpy as np
temp = np.arange(48)
shape = (4, 3, 4)
arr = temp.reshape(shape)
print(arr)

shape = (6, 8)
arr = temp.reshape(shape)
print(arr)
```
# 결과
<p align="left">
 <img src = "4week_report.3.jpg">
</p>
