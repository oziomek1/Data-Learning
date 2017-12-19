# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.
main_file_path = '../input/train.csv'
data = pd.read_csv(main_file_path)
print('\n\nData info():')
data.info()
print('\n\nData columns:\n', data.columns)

data.LotArea.plot(kind='line', color='yellow', label='Lot Area', linewidth=1, grid=True, linestyle='-',
                  figsize=(10, 10))
data.SalePrice.plot(kind='line', color='green', label='Sale Price', linewidth=1, grid=True, linestyle='-',
                    figsize=(10, 10))
plt.legend(loc='upper right')
plt.xlabel('House Id')
plt.ylabel('Lot Area')
plt.title('Line Plot')
plt.show()

data.plot(kind='scatter', x='LotArea', y='SalePrice', alpha=0.5, color=['red', 'blue'], figsize=(10, 10))
plt.legend(loc='upper right')
plt.xlabel('Lot Area')
plt.ylabel('Sale Price')
plt.title('Scatter Plot')
plt.show()

data.GarageArea.plot(kind='hist', bins=50, figsize=(10, 10), normed=True)
plt.show()

series = data['LotArea']
print(type(series))
data_frame = data[['LotArea']]
print(type(data_frame))

x = data['LotArea'] > 50000
data[x]
y = data['Alley'].notnull()
data[y]
data[(data['LotArea'] > 100000) & (data['SalePrice'] > 100000)]

for index, value in data[['LotArea']][0:2].iterrows():
    print(index, " : ", value)

list1 = [1, 2, 3, 4]
list2 = [5, 6, 7, 8]
print(list1, list2)
list3 = [i ** i for i in list1]
print(list3)
z = zip(list1, list2)
z_list = list(z)
print(z_list)

un_zip = zip(*z_list)
un_list1, un_list2 = list(un_zip)
print(un_list1, un_list2)

threshold = sum(data.LotArea) / len(data.LotArea)
print("Threashold area:", threshold)
data["average_area"] = ["huge" if i > threshold else "small" for i in data.LotArea]
data.loc[:50, ["average_area", "LotArea"]]

data = data.drop('average_area', axis=1)

# first 5 rows, last 5 rows
data.head()
# data.tail()
# data.shape # number of columns x rows
print("\n", data.describe(), "\n\n")
print(data['SalePrice'].value_counts())

new_data = data.head()
new_data
melted = pd.melt(frame=new_data, id_vars='Id', value_vars=['LotArea', 'MSSubClass'])
melted
# reverse of melt operation
melted.pivot(index='Id', columns='variable', values='value')

data.boxplot(column='SalePrice', by='MSSubClass')
plt.show()

data1 = data.head()
data2 = data.tail()
conc_data = pd.concat([data1, data2], axis=0, ignore_index=False)
conc_data

data["Alley"].value_counts(dropna=False)
data["Alley"].fillna('empty', inplace=True)

assert data["Alley"].notnull().all()

# data frames from dictionary
country = ["Spain", "France"]
population = ["11", "12"]
list_label = ["country", "population"]
list_col = [country, population]
zipped = list(zip(list_label, list_col))
data_dict = dict(zipped)
df = pd.DataFrame(data_dict)
df['capital'] = ['madrid', 'paris']
df['income'] = 0
df

data1 = data.loc[:, ['LotArea', 'GarageArea']]
data1.plot()

data1.plot(subplots=True)

fig, axes = plt.subplots(nrows=2, ncols=1)
data.plot(kind="hist", y="SalePrice", bins=50, normed=True, ax=axes[0])
data.plot(kind="hist", y="SalePrice", bins=50, normed=True, ax=axes[1], cumulative=True)

#data2 = data.head()
#date_list = ["1992-01-10", "1992-02-10", "1992-03-10", "1993-03-15", "1993-03-16"]
#datetime_object = pd.to_datetime(date_list)
#data2["date"] = datetime_object
#data2 = data2.set_index("date")
#data2
#data2.resample("M").mean().interpolate("linear")

data3 = data.set_index(["GarageCars", "GarageArea"])
data3.head(30)

filter_1 = data2.LotFrontage > 68
filter_2 = data2.OverallQual > 7
data2[filter_1 & filter_2]
def div(n):
    return n/2
# data.SalePrice.apply(div)
data.SalePrice.sort_values()
data.head(10)
data3 = pd.read_csv(main_file_path)
data3.groupby("SalePrice").mean()