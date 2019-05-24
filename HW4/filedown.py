# -*- coding:utf-8 -*-

# -*- coding:utf-8 -*-

import os
import requests
import pandas as pd
import numpy as np
import csv

rootPath = os.getcwd()
url = 'https://raw.githubusercontent.com/gupta-abhay/10-601B/master/hw4/handout/largedata/train_data.tsv'
saverpath =  rootPath + '/largedata/' + url.split('/')[-1]
print(saverpath)
response = requests.get(url)
content = response.content.decode('utf-8')
cv = csv.reader(content.splitlines(), delimiter = '\n')

print(content)

cont = list(cv)
# print(len(cont))
# file = []
for line in cont:
    print(line)
    print(line[0])
#     data = []
#     label = line[0][0]
#     value = line[0][1:].lstrip()
#     # print(label + '\t' +  value)
#     data.append(label)
#     data.append(value)
#     file.append(data)


with open(saverpath, 'w') as f:
    for line in cont:
        f.write(line[0])
        f.write('\n')


# print(cont)
# df = pd.DataFrame(cont)
# print(df.head())
# df.to_csv(saverpath, index = False, header = False)


# cr = csv.reader(content)
