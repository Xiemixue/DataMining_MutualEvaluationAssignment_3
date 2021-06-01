import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import warnings
import matplotlib.pyplot as plt
%matplotlib inline
warnings.filterwarnings("ignore")

filename = 'vgsales.csv'
vgs = pd.read_csv(filename, index_col='Rank')

print("shape={}".format(vgs.shape))
vgs.head(5)    #打印前5行


vgs.isnull().sum()    #统计缺失值


vgs = vgs.dropna()    #丢弃含有缺失值的行
vgs.isnull().sum()    #重新统计缺失值


vgs.info()


vgs.Year = vgs.Year.astype(int)


# 获取游戏类型的排序
Genre_data = vgs.groupby(['Genre']).sum().loc[:, 'Global_Sales'].sort_values(ascending = False)

# 进行画图
sns.barplot(y = Genre_data.index, x = Genre_data.values, orient='h', color="seagreen")
plt.ylabel("Genre")
plt.xlabel("Global_Sales (In Millions)")
plt.show()


# 获取游戏平台的排序
Platform_data = vgs.groupby(['Platform']).count().loc[:,"Name"].sort_values(ascending = False)

# 进行画图
plt.figure(figsize=(10,10))
sns.barplot(y = Platform_data.index, x = Platform_data.values, orient='h', color="seagreen")
plt.ylabel("Platform")
plt.xlabel("The amount of games")
plt.show()


# 获取游戏发行商的排序
Publisher_data = vgs.groupby(['Publisher']).sum().loc[:,"Global_Sales"].sort_values(ascending = False)
print("游戏发行商个数={}".format(len(Publisher_data)))

# 只保留Publisher_data.values>50的数据
Publisher_data = Publisher_data[Publisher_data.values > 100]

# 进行画图
sns.set(font_scale=1)
plt.ylabel("Publisher")
plt.xlabel("Global_Sales (In Millions)")
sns.barplot(y = Publisher_data.index, x = Publisher_data.values, orient='h', color="seagreen")
plt.show()


def lineplot(df, ylabel, title='Sales by Year', legendsize=10, legendloc='upper left'):
    year = df.index.values
    na = df.NA_Sales
    eu = df.EU_Sales
    jp = df.JP_Sales
    other = df.Other_Sales
    global_ = df.Global_Sales

    if df is count_sales_group:
        region_list = [na, eu, jp, other, global_]
        columns = ['NA', 'EU', 'JP', 'OTHER', 'WORLD WIDE']
    else:
        region_list = [na, eu, jp, other, global_]
        columns = ['NA', 'EU', 'JP', 'OTHER', 'WORLD WIDE']

    for i, region in enumerate(region_list):
        plt.plot(year, region, label=columns[i])

    plt.ylabel(ylabel)
    plt.xlabel('Year')
    plt.title(title)
    plt.legend(loc=legendloc, prop={'size': legendsize})
    plt.show()
    plt.clf()


years = [2017, 2020]
total_sales_group = vgs.groupby(['Year']).sum().drop(years)
average_sales_group = vgs.groupby(['Year']).mean().drop(years)
count_sales_group = vgs.groupby(['Year']).count().drop(years)

lineplot(total_sales_group, title='Sales by Year', ylabel="Sales (In Millions)", legendsize=8)



lineplot(average_sales_group, title = 'Average Sales per Game per Year', ylabel ='Sales (In Millions)',
         legendsize = 8, legendloc = 'upper right')


lineplot(count_sales_group, title = 'Games Counts by Year', ylabel ='Count',
         legendsize = 8, legendloc = 'upper left')


from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image

stopwords = set(STOPWORDS)

for x in vgs.Genre.unique():
    wc = WordCloud(background_color="white", max_words=2000,
                   stopwords=stopwords, max_font_size=40, random_state=42)
    wc.generate(vgs.Name[vgs.Genre == x].to_string())
    plt.imshow(wc)
    plt.title(x)
    plt.axis("off")
    plt.show()


# 获取销售额数据
sales = vgs[['Year', 'Global_Sales']].dropna()
sales = sales.groupby(by = ['Year']).sum().reset_index()
print(sales)

from sklearn.preprocessing import LabelEncoder

# categorical_labels = ['Platform','Genre','Publisher','NA_Sales','EU_Sales','JP_Sales']
categorical_labels = ['Platform', 'Genre', 'Publisher']
numerical_lables = ['Global_Sales']
enc = LabelEncoder()
encoded_df = pd.DataFrame(columns=['Platform', 'Genre', 'Publisher', 'Global_Sales'])

for label in categorical_labels:
    temp_column = vgs[label]
    encoded_temp_col = enc.fit_transform(temp_column)
    encoded_df[label] = encoded_temp_col

for label in numerical_lables:
    encoded_df[label] = vgs[label].values

print(encoded_df.head())


from sklearn.model_selection import train_test_split
train, test = train_test_split(encoded_df, test_size=0.1, random_state=1)

def data_splitting(df):
    x=df.drop(['Global_Sales'], axis=1)
    y=df['Global_Sales']
    return x, y

x_train, y_train = data_splitting(train)
x_test, y_test = data_splitting(test)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
log = LinearRegression()
log.fit(x_train , y_train)
y_pred = log.predict(x_test)
n = len(x_test)
p = x_test.shape[1]
r2_value = r2_score(y_test, y_pred)
adjusted_r2_score = 1 - (((1-r2_value)*(n-1)) /(n-p-1))
print("r2_score for Linear Reg model : ",r2_score(y_test,y_pred))
print("adjusted_r2_score Value       : ",adjusted_r2_score)
print("MSE for Linear Regression     : ",mean_squared_error(y_test,y_pred))


from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=200,min_samples_split=20,random_state=43)
rf_model.fit(x_train,y_train)
y_pred = rf_model.predict(x_test)
n = len(x_test)
p = x_test.shape[1]
r2_value = r2_score(y_test,y_pred)
adjusted_r2_score = 1 - (((1-r2_value)*(n-1)) /(n-p-1))
print("r2_score for Random Forest Reg model : ",r2_score(y_test,y_pred))
print("adjusted_r2_score Value              : ",adjusted_r2_score)
print("MSE for Random Forest Regression     : ",mean_squared_error(y_test,y_pred))