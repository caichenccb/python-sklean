## -*- coding: utf-8 -*-
#"""
#Created on Mon Jul  8 21:37:28 2019
#
#@author: 92156
#"""
#
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns
from sklearn import datasets,linear_model,discriminant_analysis,cross_validation,model_selection
from scipy import stats
from sklearn.cross_validation import train_test_split
sns.set_style("whitegrid") 
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
cc1=pd.read_excel("作业1.xlsx")
my_font = font_manager.FontProperties(fname="C:\Windows\Fonts\msyh.ttc")
#b=0;c=0
#q=[]
#ss=cc1[["性别","社会支持总分"]]
#ss1 = ss.set_index('性别')
#b=ss1.ix[1]
#c=ss1.ix[2]
#
#b.columns=["1"]
#c.columns=["2"]
#
#frames=[b,c]
#result = pd.concat(frames)
#
##res =pd.join(b,c)
#aaa=list(set(cc1["性别"]))


#q=[]
#ss=ss.tolist()
#ss.sort()
##a=set(ss,key=ss.index)
#list3=list(set(ss))
#score_list_int = list(map(int,list3))
#
#b=len(ss)
#for i in range(b):
#    if ss[i]==ss[i-1]:
##        print("gg")
#        c=c+1
#        cc1[ss[i]]=c
#    else:
#        c=1
##        print("bb")
#aaa=cc1[list3]
#ssss=aaa.iloc[0,:]
#ssssss=ssss.to_list()
#labels=score_list_int
##xx=cc1[c,b]
#plt.pie(ssssss,labels=labels,shadow=True, autopct="%0.2f%%")
##ss=cc1["社会支持总分"]
##ss=ss.tolist()
##data=ss
#fig,axes=plt.subplots(1,2,sharey=True)
#data=result
#sns.distplot(data)   #  data 是dataframe 数组 color  ，age是列
#sns.boxplot(data=data,palette="Set3") #中图
##
#result.hist(bins=40)
#plt.title("2",fontproperties=my_font)
#plt.axis('equal')
#plt.legend(loc="upper right",fontsize=10,bbox_to_anchor=(1.1,1.05),borderaxespad=0.3)
#fig = plt.figure()
#fig.tight_layout(pad=0.4, w_pad=3.0, h_pad=3.0)
#g = sns.FacetGrid(cc1, col="性别", hue="社会支持总分",sharex=True, sharey=True)# 都共享
#g.map(plt.bar, "年龄", "年级号", alpha=0.8)
#g.add_legend();


## 饼图
#import matplotlib.pyplot as plt
#from matplotlib import font_manager
#import seaborn as sns
#plt.rcParams['font.sans-serif'] = ['SimHei']
#plt.rcParams['axes.unicode_minus'] = False
#cc1=pd.read_excel("作业1.xlsx")
#my_font = font_manager.FontProperties(fname="C:\Windows\Fonts\msyh.ttc")
#b=0;c=0
#ss=cc1["年级号"]
#q=[]
#ss=ss.tolist()
#ss.sort()
##a=set(ss,key=ss.index)
#list3=list(set(ss))
#score_list_int = list(map(int,list3))
#
#b=len(ss)
#for i in range(b):
#    if ss[i]==ss[i-1]:
##        print("gg")
#        c=c+1
#        cc1[ss[i]]=c
#    else:
#        c=1
##        print("bb")
#aaa=cc1[list3]
#ssss=aaa.iloc[0,:]
#ssssss=ssss.to_list()
#labels=score_list_int
##xx=cc1[c,b]
#explode = (0, 0.1,0.1,0.1,0.1)
#x_0 = [1,0,0,0]#用于显示空心
#colors=["red","blue","grey","green","yellow"]
#plt.pie(ssssss,radius=1.0,pctdistance = 0.8,labels=labels,colors=colors,startangle=90,autopct='%1.1f%%')
##plt.pie(x_0, radius=0.6,colors = 'w')
###ss=cc1["社会支持总分"]
###ss=ss.tolist()
###data=ss
###sns.distplot(data) #中图
##
###ss.hist(bins=40,facecolor="black",edgecolor="blue")
#plt.title("年级号-饼图",fontproperties=my_font)
#plt.axis('equal')
#plt.legend(loc="upper right",fontsize=10,radius=[40,75],bbox_to_anchor=(1.1,1.05),borderaxespad=0.3)
#fig = plt.figure()
#fig.tight_layout(pad=0.4, w_pad=3.0, h_pad=3.0)
#
##箱形图
#import pandas as pd
#import numpy as np 
#import matplotlib.pyplot as plt
#from matplotlib import font_manager
#import seaborn as sns
#sns.set(style='whitegrid', color_codes=True)
#plt.rcParams['font.sans-serif'] = ['SimHei']
#plt.rcParams['axes.unicode_minus'] = False
#cc1=pd.read_excel("作业1.xlsx")
#my_font = font_manager.FontProperties(fname="C:\Windows\Fonts\msyh.ttc")
#b=0;c=0
#q=[]
#ss=cc1[["年级号","社会支持总分"]]
#ss1 = ss.set_index('年级号')
#b=ss1.ix[1]
#c=ss1.ix[2]
#d=ss1.ix[3]
#e=ss1.ix[4]
#f=ss1.ix[5]
#
#b.columns=["1"]
#c.columns=["2"]
#d.columns=["3"]
#e.columns=["4"]
#f.columns=["5"]
#frames=[b,c,d,e,f]
#result = pd.concat(frames)
#
##res =pd.join(b,c)
##aaa=list(set(cc1["年级号"]))
#
#
##q=[]
##ss=ss.tolist()
##ss.sort()
###a=set(ss,key=ss.index)
##list3=list(set(ss))
##score_list_int = list(map(int,list3))
##
##b=len(ss)
##for i in range(b):
##    if ss[i]==ss[i-1]:
###        print("gg")
##        c=c+1
##        cc1[ss[i]]=c
##    else:
##        c=1
###        print("bb")
##aaa=cc1[list3]
##ssss=aaa.iloc[0,:]
##ssssss=ssss.to_list()
##labels=score_list_int
###xx=cc1[c,b]
##plt.pie(ssssss,labels=labels,shadow=True, autopct="%0.2f%%")
###ss=cc1["社会支持总分"]
###ss=ss.tolist()
###data=ss
##fig,axes=plt.subplots(1,2,sharey=True)
#data=result
#sns.boxplot(data=data,palette="Set3",linewidth = 2, #线宽
#            width = 0.8, #箱之间的间隔比例
#            fliersize = 3, #异常点大小
#             #设置调色板
#            whis = 1.5,     #设置IQR
#            notch = True,   #设置是否以中值做凹槽
#            #筛选类别
#) #中图
##
###ss.hist(bins=40,facecolor="black",edgecolor="blue")
##plt.title("年级号-饼图",fontproperties=my_font)
##plt.axis('equal')
#plt.legend(loc="upper right",fontsize=10,bbox_to_anchor=(1.1,1.05),borderaxespad=0.3)
#fig = plt.figure()
#fig.tight_layout(pad=0.4, w_pad=3.0, h_pad=3.0)
#plt.show()
#


#数据清洗
#cc1=pd.read_csv("cc1.csv")
#cc2=pd.read_csv("cc2.csv")
#cc3=pd.read_csv("cc3.csv",encoding="gbk")
#cc4=pd.read_csv("cc4.csv",encoding="gbk")
#cc=cc3.dropna(subset=["sf"])
#
#bb=pd.merge(cc,cc2,on=["spbm","je","dtime"])
#bb=bb[["djh_x","sj_x","spbm","sl_x","je","dtime","syjh_x"]]
#bb.columns=["djh","sj","spbm","sl","je","dtime","syjh"]
#
#cc1["id1"]=1
#cc2["id2"]=2
#cc3["id3"]=3
#cc4["id4"]=4
#
##删除
#
##cc1的操作
#
##删除csny错误值
#aa=cc1.dropna()
#aa["bijiao"]=aa["csny"]>aa["djsj"]
#aa=aa[~aa['bijiao'].isin([True])] # 删除里面数值为TRUE的 
#aa["csny"]=pd.to_datetime(aa["csny"])
#aa["djsj"]=pd.to_datetime(aa["djsj"])
#qq=aa[aa['bijiao'].isin([True])]
#aa["b"]=aa["csny"].dt.year==aa["djsj"].dt.year
#aa["x"]=(abs(aa["csny"].dt.year-aa["djsj"].dt.year))>70
#aa=aa[~aa['x'].isin([True])]
#aa=aa[~aa['b'].isin([True])]
##删除重复值
#aaaa=aa.duplicated()
#
#
#
##cc2的操作
#bbbb=bb.duplicated() # 查重
#bbbb=bb.drop_duplicates() #删重
#bbbb["jihao"]=bbbb["sl"]==-1
#aa=bbbb["sl"]==-1
#data=bbbb[bbbb["djh"].str.contains(r"[\D]")]  #处理异常
#qqq=data[data["djh"].str.contains("E")]
#data=data.drop(qqq.index)
#qq=data[data["spbm"].str.contains("E")]
#data=data.drop(qq.index)
#ww=bbbb[bbbb['jihao'].isin([True])]
#data["x"]=data["je"]==0
#data=data[~data['x'].isin([True])]         #删除je 为0的
#
#
#
#
##cc3的操作
#cc3=cc3.dropna(subset=["sf"])
#cc3["x"]=cc3["je"]==0
#cc3=cc3[~cc3['x'].isin([True])]         #删除je 为0的
#
#
#
##cc4的操作
#cc4=cc4.dropna()
#cc4["x"]=cc4["sj"]==0
#cc4=cc4[~cc4['x'].isin([True])]         #删除je 为0的
#
##替代
##
##cc1的操作
##
##djsj的操作
#aa=cc1
#print(aa["djsj"].isnull().any())
#aa["djsj"]=pd.to_datetime(aa["djsj"])
#aa=aa.dropna(subset=["djsj"])
#aa["djsj"].dt.year.mean()
#aa["djsj"].dt.month.mean()
#aa["djsj"].dt.day.mean()
#aa["djsj"]=aa["djsj"].apply(str)
#cc1["djsj"]=cc1["djsj"].fillna("2012-06-15 00:00:00")
#
##csny的操作
#aa=cc1
#aa["bijiao"]=aa["csny"]>aa["djsj"]
#ss=aa[aa['bijiao'].isin([True])] # 删除里面数值为不是True的 
#aa=aa[~aa['bijiao'].isin([True])] # 删除里面数值为TRUE的 
#aa["csny"]=pd.to_datetime(aa["csny"])
#aa["djsj"]=pd.to_datetime(aa["djsj"])
#qq=aa[aa['bijiao'].isin([True])]
#aa["b"]=aa["csny"].dt.year==aa["djsj"].dt.year
#aa["x"]=(abs(aa["csny"].dt.year-aa["djsj"].dt.year))>70
#ww=aa[aa['x'].isin([True])]
#ee=aa[aa['b'].isin([True])]
#aa=aa[~aa['x'].isin([True])]
#aa=aa[~aa['b'].isin([True])]
#
#aa["csny"].dt.year.mean()
#aa["csny"].dt.month.mean()
#aa["csny"].dt.day.mean()  #median()  分位数
#
#aa["csny"]=aa["csny"].fillna("1976/06/15 00:00:00")
#ww["csny"]="1976/06/15 00:00:00"
#ee["csny"]="1976/06/15 00:00:00"
#ss["csny"]="1976/06/15 00:00:00"
#aa=pd.concat([aa,ww,ee,ss])
#
##cc2的操作
#
#bb=bb[bb["djh"].str.contains(r"[\D]")]
#bb=bb[~bb["djh"].str.contains("E")]  #~ 表示否
#bb["x"]=bb["je"]==0
#ss=bb.loc[bb["x"] == True]
#ss["je"]=bb["je"].mean()
#
##cc3的操作
#
#cc3=cc3.dropna(subset=["sf"])
#cc3["x"]=cc3["je"]==0
#ss=cc3[cc3['x'].isin([True])] 
#cc3=cc3[~cc3['x'].isin([True])] 
#ss["je"]=cc3["je"].mean()
#cc3=pd.concat([ss,cc3])
#
##cc4的操作
#
#cc4=cc4.drop_duplicates()
#cc4["x"]=cc4["je"]==0
#ss=cc4[cc4['x'].isin([True])] 
#cc4=cc4[~cc4['x'].isin([True])] 
#ss["je"]=cc4["je"].mean()
#cc4=pd.concat([ss,cc4])
#




#处理极端值 用分位数替代
#凡小于百分之1分位数和大于百分之99分位数的值将会被百分之1分位数和百分之99分位数替代：
def cc(x,quantile=[0.01,0.99]):
#    """盖帽法处理异常值
#    Args：
#        x：pd.Series列，连续变量
#        quantile：指定盖帽法的上下分位数范围
#    """

# 生成分位数
    Q01,Q99=x.quantile(quantile).values.tolist()

# 替换异常值为指定的分位数
    if Q01 > x.min():
        x=x.copy()
        x.loc[x<Q01] = Q01
    if Q99 < x.max():
        x = x.copy()
        x.loc[x>Q99] = Q99
    return(x)
    
    
#第三次作业
#线性回归和相关性
    
#相关性
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import *
data = pd.read_excel('财政收入.xls')
xx = data.corr()          #相关性
#换下位子
mid = xx['y']
xx.drop(labels=['y'], axis=1,inplace = True)
xx.insert(4, 'y', mid)
#排个序
xx=xx.sort_index(axis = 0,ascending = True,by = 'y')
#画个图
plot(xx.columns,xx["y"],color="blue", linewidth=2.5)

#检验 显著性检验
import scipy.stats as stats  
x=data["x1"]
y=data["y"]
r, p=stats.pearsonr(x,y)
#p: 3.640788395143971e-08   非常显著
x2=data["x3"]
r2, p2=stats.pearsonr(x2,y)
#p2: 2.816413632485161e-17
x1=data["x2"]
r1, p1=stats.pearsonr(x1,y)
#p1:2.0258026252456704e-08
x3=data["x4"]
r3, p3=stats.pearsonr(x3,y)
#p3:6.222728917601221e-10

#线性回归
#seaborn 做的不一定好
import seaborn as sns 
import pandas as pd
data = pd.read_excel('财政收入.xls')
sns.pairplot(data, x_vars=['x1','x2','x3',"x4"], y_vars='y',diag_kind="kde", markers=".",
              plot_kws=dict(s=50, edgecolor="b", linewidth=1),
                 diag_kws=dict(shade=True));
             
#用statsmodels库来做          
import  pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
data = pd.read_excel('财政收入.xls')
y = data['y']
x1 = data['x1']
x2 = data['x2']
x3 = data['x3']
x4 = data["x4"]
x = np.column_stack((x1, x2,x3,x4))#表示自变量包含x1和x1的平方，如果需要更高次，可再做添加，也可以添加交叉项x1*y，或者其他变量
#注意自变量设置时，每项高次都需依次设置,x1**2表示乘方，或者x1*x1也可以
x = sm.add_constant(x) #增一列1，用于分析截距
model = sm.OLS(y, x)
#数据拟合，生成模型
result = model.fit()
print(result.summary())
#应用模型预测
Y=result.fittedvalues
#     OLS Regression Results                            
#==============================================================================
#Dep. Variable:                      y   R-squared:                       0.997
#Model:                            OLS   Adj. R-squared:                  0.996
#Method:                 Least Squares   F-statistic:                     1467.
#Date:                Thu, 25 Jul 2019   Prob (F-statistic):           2.47e-21
#Time:                        18:34:08   Log-Likelihood:                -7.5480
#No. Observations:                  22   AIC:                             25.10
#Df Residuals:                      17   BIC:                             30.55
#Df Model:                           4                                         
#Covariance Type:            nonrobust                                         
#==============================================================================
#                 coef    std err          t      P>|t|      [0.025      0.975]
#------------------------------------------------------------------------------
#const       1937.8648      6.967    278.151      0.000    1923.166    1952.564
#x1            -0.0013      0.000     -3.689      0.002      -0.002      -0.001
#x2             0.0014      0.000      3.656      0.002       0.001       0.002
#x3             0.0008      0.000      7.234      0.000       0.001       0.001
#x4          4.554e-05   1.31e-05      3.487      0.003     1.8e-05    7.31e-05
#==============================================================================
#Omnibus:                        0.880   Durbin-Watson:                   1.830
#Prob(Omnibus):                  0.644   Jarque-Bera (JB):                0.229
#Skew:                           0.239   Prob(JB):                        0.892
#Kurtosis:                       3.149   Cond. No.                     1.76e+07

#方差分析    https://blog.csdn.net/weixin_41869644/article/details/89854657
import seaborn as sns 
import pandas as pd
data = pd.read_excel('财政收入.xls')
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

formula = 'y~ x1 + x2+x3+x4'
anova_results = anova_lm(ols(formula,data).fit())
print(anova_results)
#因为两个因素的P值都小于0.05，拒绝原假设，说明时间和浓度对实验结果有显著影响。
#使用tukey方法分别对浓度和时间进行多重比较。
#分对象进行比较 男女
#print(pairwise_tukeyhsd(data['y'], data['x1']))
#print(pairwise_tukeyhsd(data['y'], data['x2']))
#print(pairwise_tukeyhsd(data['y'], data['x3']))
#print(pairwise_tukeyhsd(data['y'], data['x4']))

#阴性率与儿童年龄
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import *
data = pd.read_excel('阴性率与儿童年龄.xls')
xx = data.corr()          #相关性
#结果是 ： 0.8456641

#显著性检验
import scipy.stats as stats  
x=data["阴性率"]
y=data["年龄"]
r, p=stats.pearsonr(x,y)
#显著性为0.016509568788826937   显著

#线性回归
#seaborn 做的不一定好
import seaborn as sns 
import pandas as pd
data = pd.read_excel('阴性率与儿童年龄.xls')
sns.pairplot(data, x_vars=["阴性率"], y_vars='年龄',diag_kind="kde", markers=".",
              plot_kws=dict(s=50, edgecolor="b", linewidth=1),
                 diag_kws=dict(shade=True));
             
             
#用statsmodels库来做                  
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
data = pd.read_excel('阴性率与儿童年龄.xls',encoding="gbk")
y = data['年龄']
x1 = data['阴性率']
x = sm.add_constant(x1)#添加截距项#,回归方程添加一列x0=1
model = sm.OLS(y, x)
#数据拟合，生成模型
results = model.fit()
print(results.summary())
#                   OLS Regression Results                            
#==============================================================================
#Dep. Variable:                     年龄   R-squared:                       0.715
#Model:                            OLS   Adj. R-squared:                  0.658
#Method:                 Least Squares   F-statistic:                     12.55
#Date:                Fri, 26 Jul 2019   Prob (F-statistic):             0.0165
#Time:                        11:59:51   Log-Likelihood:                -10.389
#No. Observations:                   7   AIC:                             24.78
#Df Residuals:                       5   BIC:                             24.67
#Df Model:                           1                                         
#Covariance Type:            nonrobust                                         
#==============================================================================
#                 coef    std err          t      P>|t|      [0.025      0.975]
#------------------------------------------------------------------------------
#const         -6.5434      3.014     -2.171      0.082     -14.291       1.204
#阴性率            0.1219      0.034      3.543      0.017       0.033       0.210
#==============================================================================
#Omnibus:                          nan   Durbin-Watson:                   0.677
#Prob(Omnibus):                    nan   Jarque-Bera (JB):                0.422
#Skew:                           0.262   Prob(JB):                        0.810
#Kurtosis:                       1.917   Cond. No.                         553.
#==============================================================================


#linear_model 模型
from matplotlib import pyplot as plt
from sklearn import datasets,linear_model,discriminant_analysis,cross_validation,model_selection
from matplotlib import pyplot as plt
x=np.array(data['阴性率']).reshape(7,1)
y=np.array(data['年龄']).reshape(7,1)

model=linear_model.LinearRegression()

model.fit(x,y)

coef=model.coef_ #获取自变量系数

model_intercept=model.intercept_#获取截距

R2=model.score(x,y) #R的平方

print('线性回归方程为：','\n','y=’{}*x+{}'.format(coef,model_intercept))

#scipy.polyfit 模型
import scipy
def fit_linear_model(x,y,slope=None):
    '''Linear least squares (LSQ) linear regression (polynomial fit of degree=1)
    Returns:
        m (float) = slope of linear regression line of form (y = m*x + b)
        b (float) = intercept of linear regression line'''

    assert len(x)==len(y), ("Arrays x & Y must be equal length to fit "
                            "linear regression model.")
    if slope == None:
        (m,b) = scipy.polyfit(x,y,deg=1)
    else:
        LSQ = lambda b: np.sum( (y-(slope*x+b))**2.0 )
        res = scipy.optimize.minimize(LSQ,x0=1,bounds=None)
        (m,b) = (slope,res.x[0])
    return (m,b)
#np.polyval 模型
poly = np.polyfit(x,y,deg=1)
x=np.array(data['阴性率'])
y=np.array(data['年龄'])

poly = np.polyfit(x,y,deg=1)

z = np.polyval(poly, x)
plt.plot(x, y, 'o')
plt.plot(x, z)

plt.show()
poly


#scipy 中的stats库
import scipy
fit_linear_model(x,y)
from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
print("r-squared:", r_value**2)
p_value
slope


pd.plotting.scatter_matrix(data[["阴性率","年龄"]],alpha=0.8, figsize=(10, 10), diagonal='kde')



from scipy import stats
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_excel("阴性率与儿童年龄.xls",sheetname=0)
x=df['阴性率']
y=df['年龄']
#画出x与y的散点图
plt.scatter(x,y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('the linear regression')
#线性回归拟合
x_m=np.mean(x)
y_m=np.mean(y)
x1=(x-x_m)
y1=y-y_m
x2=sum((x-x_m)**2)
xy=sum(x1*y1)
#回归参数的最小二乘估计
beta1=xy/x2
beta0=y_m-beta1*x_m
#输出线性回归方程
print('y=',beta0,'+',beta1,'*x')
#画出回归方程的函数图
#方差
sigma2=sum((y-beta0-beta1*x)**2)/(18)
#标准差
sigma=np.sqrt(sigma2)
#求t值
t=beta1*np.sqrt(x2)/sigma
print('t=',t)
#已知临界值求p值
p=stats.t.sf(t,18)
print('p=',p)
 
#输出检验结果
if p<0.05:
    print ('the linear regression between x and y is significant')
else:
	print('the linear regression between x and y is not significant')
 
plt.show()


#第四次作业 
#聚类分析和主成分分析
from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
df=pd.read_excel("工业排污.xls",encoding = 'gbk')

def get_boxplot(data, start, end):
    fig, ax = plt.subplots(1, end-start, figsize=(24, 4))
    for i in range(start, end):
        sns.boxplot(y=data[data.columns[i]], data=data, ax=ax[i-start])
get_boxplot(df, 1,4 )
def drop_outlier(data, start, end):
    for i in range(start, end):
        field = data.columns[i]
        Q1 = np.quantile(data[field], 0.25)
        Q3 = np.quantile(data[field], 0.75)
        deta = (Q3 - Q1) * 1.5
        data = data[(data[field] >= Q1 - deta) & (data[field] <= Q3 + deta)]
    return data
del_df = drop_outlier(df, 1, 4)
print("原有样本容量:{0}, 剔除后样本容量:{1}".format(df.shape[0], del_df.shape[0]))
get_boxplot(del_df, 1, 4)

km = KMeans(n_clusters=2, random_state=10)
km.fit(del_df[["工业废气排放总量","工业废水排放总量","二氧化硫排放总量"]])
print(km.cluster_centers_) 
print(km.labels_)
del_df["label"]=km.labels_
del_df["预测"]=km.predict(del_df[["工业废气排放总量","工业废水排放总量","二氧化硫排放总量"]])

#看当k取何值时分类较好
import matplotlib.pyplot as plt
K = range(1, 10)
sse = []
for k in K:
    km = KMeans(n_clusters=k, random_state=10)
    km.fit(del_df[["工业废气排放总量","工业废水排放总量","二氧化硫排放总量"]])
    sse.append(km.inertia_)
plt.figure(figsize=(8, 6))
plt.plot(K, sse, '-o', alpha=0.7)
plt.xlabel("K")
plt.ylabel("SSE")
plt.show()


#当k为3时，看上去簇内离差平方和之和的变化已慢慢变小，那么，我们不妨就将球员聚为7类。如下为聚类效果的代码：
#肘式
km = KMeans(n_clusters=3, random_state=10)
km.fit(del_df[["工业废气排放总量","工业废水排放总量","二氧化硫排放总量"]])
print(km.cluster_centers_) 
print(km.labels_)
del_df["label"]=km.labels_
del_df["预测"]=km.predict(del_df[["工业废气排放总量","工业废水排放总量","二氧化硫排放总量"]])
sns.barplot(x=del_df.index, y="label", data=del_df,  palette="Set3")




#层次聚类
import pandas as pd
import seaborn as sns  #用于绘制热图的工具包
from scipy.cluster import hierarchy  #用于进行层次聚类，话层次聚类图的工具包
from scipy import cluster   
import matplotlib.pyplot as plt
from sklearn import decomposition as skldec #用于主成分分析降维的包
Z = hierarchy.linkage(del_df[["工业废气排放总量","工业废水排放总量","二氧化硫排放总量"]], method ='ward',metric='euclidean')
hierarchy.dendrogram(Z,labels = del_df.index)

#利用sns 聚类
sns.clustermap(del_df[["工业废气排放总量","工业废水排放总量","二氧化硫排放总量"]],method ='ward')   # ’single 最近点算法。   ’complete   这也是最远点算法或Voor Hees算法   ’average   UPGMA算法。 ’weighted  （也称为WPGMA  ’centroid  WPGMC算法。


#主成分分析
import pandas as pd 

X=pd.read_excel("消费结构.xls")
from sklearn.decomposition import PCA
 
pca = PCA(n_components=2)
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
X_new = pca.fit_transform(X)
plt.scatter(X_new[:, 0], X_new[:, 1],marker='o')
plt.show()
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
X_new = pca.transform(X)
plt.plot(X_new[:, 0],marker='o')
y=X.index
red_x=[]
red_y=[]
for i in range(len(X_new)):
    red_x.append(X_new[i][0])
    red_y.append(X_new[i][1])
plt.scatter(red_x,red_y,c="r",marker="x")
    
