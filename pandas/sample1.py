import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt

# Object Creation
s = pd.Series([1,3,5,np.nan,6,8])
print(s)

# freq M, D, H
dates = pd.date_range('20130101', periods=6, freq="D")
print(dates)

df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))

print(df)

df2 = pd.DataFrame({'A':1.,
                    'B':pd.Timestamp('20130102'),
                    'C':pd.Series(1,index=list(range(4)),dtype='float32'),
                    'D':np.array([3] * 4, dtype='int32'),
                    'E':pd.Categorical(["test","train","test","train"]),
                    'F':'foo'})
print(df2)

# get data types
print(df2.dtypes)

# Viewing Data

print(df.head())

print(df.tail(3))

print(df.head(1))

print("----")
print(df.index)

print("----")
print(df.columns)

print("----")
print(df.values)

# get data 
# check up after about how to using described data
print("----")
print(df.describe())

print("----")
print(df.T)


# check up after about arguments
print("----")
print(df.sort_index(axis=1, ascending=False))

print("----")
print(df.sort_values(by='B', ascending=False))

# Selection
print("Selection")
print(df['A'])

# Getting

print("----")
print(df[0:3])

# Selection by Label
print("----")
print(df.loc[dates[0]]) # select by date(index label)

print("----")
print(df.loc[:,['A', 'B']])
print(df.loc[dates[0], ['A']])
print("----")
print(df.loc['20130102':'20130104', ['A','B']])
print(df.loc['20130102',['A','B']]) 

# acces scaler value
print(df.loc[dates[0],'A'])

# fast!
print(df.at[dates[0],'A'])


# Selection by Positin
print(df.iloc[3])
print(df.iloc[3:5])
print(df.iloc[3:5, 0:2])
print(df.iloc[[1,2,4], [0,3]])
print(df.iloc[1:3, :])
print(df.iloc[1,1])
print(df.iat[1,1])

# Boolean Indexing
print(df[df.B > 0])
print(df[df > 0])
df2 = df.copy()

df2['E'] = ['one','one','two','three','four','three']
print(df2)

print("----")
print(df2[df2['E'].isin(['two','four'])])

# Setting
print("-----")
s1 = pd.Series([1,2,3,4,5,6],index=pd.date_range('20130102',periods=6))
print(s1)

df['F'] = s1
print(df)

df.at[dates[0],'A'] = 0
df.iat[0,1] = 0

print(df)

df.loc[:,'D'] = np.array([5] * len(df))

print(df)
