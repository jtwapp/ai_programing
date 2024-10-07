```python
# 과제1
import numpy as np
import pandas as pd

path = 'https://github.com/dongupak/DataML/raw/main/csv/'
file = path+'vehicle_prod.csv'
df = pd.read_csv(file, index_col=0)
avgs = df.select_dtypes(np.number).mean().rename('Average')
new_df = pd.concat([df, pd.DataFrame(avgs).transpose()])
print(new_df)
```
# 결과
<p align="left">
 <img src = "6week_report1.jpg">
</p>

```python

```python
# 과제1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Path = "http://github.com/dongupak/DataML/raw/main/csv/"
File = Path + "weather.csv"
weather = pd.read_csv(File, encoding='CP949')
print(weather)
weather['month'] = pd.DatetimeIndex(weather['일시']).month

monthly = [None for x in range(12)]
monthly_wind = [0 for x in range(12)]
for i in range(12):
  monthly[i] = weather[weather['month'] == i + 1]
  monthly_wind[i] = monthly[i]['평균풍속'].mean()

months = np.arange(1, 13)
plt.bar(months, monthly_wind, color='green')
plt.xlabel('Month')
plt.ylabel('Wind')
```
# 결과
<p align="left">
 <img src = "7week_report1.jpg">
  <img src = "7week_report2.jpg">
</p>

```python

