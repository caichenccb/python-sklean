## -*- coding: utf-8 -*-
#"""
#Created on Fri Jul  5 20:54:46 2019
#
#@author: 92156
#"""
#
#import pandas as pd 
#import numpy as np
#import matplotlib 
#from matplotlib import font_manager
#import matplotlib.style as ps1
#import matplotlib.pyplot as plt
##my_font = font_manager.FontProperties(fname="C:\Windows\Fonts\msyh.ttc")
##fontproperties=my_font
#
###图表类型
####['bmh', 'classic', 'dark_background', 'fast', 
###'fivethirtyeight', 'ggplot', 'grayscale', 
###'seaborn-bright', 'seaborn-colorblind', 
###'seaborn-dark-palette', 'seaborn-dark', '
###seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted', 
###'seaborn-notebook', 'seaborn-paper', 'seaborn-pastel'
###, 'seaborn-poster', 'seaborn-talk', 'seaborn-ticks', 
###'seaborn-white', 'seaborn-whitegrid', 'seaborn',
### 'Solarize_Light2', '_classic_test']
##
###税
#def tax(salary_sum):
#    if salary_sum <=3500:
#        tax =0
#    elif salary_sum <= 5000:
#        tax=0.03*(salary_sum-3500)
#    elif salary_sum <=8500:
#        tax=1500*0.3+0.1*(salary_sum-5000)
#    elif salary_sum <= 13000:
#        tax=1500*0.3+0.1*3000+(salary_sum-8000)*0.2
#    elif salary_sum <=29000:
#        tax=1500*0.3+0.1*3000+4500*0.2+(salary_sum-1500-3500-3000-4500)*0.25
#    elif salary_sum<=49000:
#        tax=1500*0.3+0.1*3000+4500*0.2++26000*0.25(salary_sum-1500-3500-3000-4500-26000)*0.3
#    
#    print (tax)
#    return(salary_sum)
#
###随机奖金
#def bonus(b_avg):
#    return pd.Series(list((np.random.normal(loc=b_avg,scale=200,size=120))))
#
#ps1.use("seaborn-deep")
#plt.grid()
#my_font = font_manager.FontProperties(fname="C:\Windows\Fonts\msyh.ttc")
#plt.title("奖金随机数",fontproperties=my_font) # 标题
#plt.hist(bonus(1500),bins=30)
#    
###五险一金
#def insurance(salary):
#    if salary<=21396:
#        return salary*0.175
#    else:
#        return 3744.58
#
###构建净收入
#def final_income (s,b_avg):
#    df_i=pd.DataFrame({
#            "月薪":[s for i in range(120)],
#            "五险一金":[insurance(s) for i in range(120)],
#            "奖金":bonus(b_avg)
#            })
#    df_i["计税部分"]=df_i["月薪"]+df_i["奖金"]
#    df_i["个人所得税"]=(df_i["计税部分"]-8500)*0.2+345
##lambda函数通不了   
## df_i["个人所得税"]=df_i["计税部分"].apply(lambda x :tax(x))
#    df_i["月收入"]=df_i["月薪"]+df_i["奖金"]-df_i["五险一金"]-df_i["个人所得税"]
#    return df_i
#result=final_income(10000,1500)
#
#
##支出模型
#
##基本生活
#gener_expense=pd.Series(np.random.randint(3000,3501,size=120))
#plt.title("基本生活支出",fontproperties=my_font)
#plt.hist(gener_expense,bins=30)
#
##购物支出
#gener_expense=pd.Series(np.random.normal(loc=5000,scale=500,size=120))  #loc  正太分布区间  ，scale 误差值
#plt.title("购物支出",fontproperties=my_font)
#plt.hist(gener_expense,bins=30)
#
##学习支出
#study=pd.Series(np.random.randint(100,500,size=120))
#plt.title("学习支出",fontproperties=my_font)
#plt.hist(study,bins=120)
#
##其他支出
#other=pd.Series(np.random.normal(loc=500,scale=40,size=120))  #loc  正太分布区间  ，scale 误差值
#plt.title("其他支出",fontproperties=my_font)
#plt.hist(other,bins=30)
#
##总支出=基本生活+购物+娱乐+学习+其他
#def final_expense():
#    df_i=pd.DataFrame({
#            "jbshzc":np.random.randint(3000,3501,size=120),
#            "gwzc":np.random.normal(loc=5000,scale=500,size=120),
#            "ylzc":np.random.randint(400,1200,size=120),
#            "xxzc":np.random.randint(100,500,size=120),
#            "qtzc":np.random.normal(loc=500,scale=40,size=120)})
##    jbshzc:基本生活支出，gwzc：购物支出，ylzc：娱乐支出，xxzc：学习支出，qtzc：其他支出
#    df_i["yzzc"]=df_i["jbshzc"]+df_i["gwzc"]+df_i["ylzc"]+df_i["xxzc"]+df_i["qtzc"]
#    return df_i
#result=final_expense()
#result[["jbshzc","gwzc","ylzc","xxzc","qtzc"]].iloc[:12].plot(kind="bar",figsize=(12,4),stacked=True,colormap="Reds")
#plt.title("总支出",fontproperties=my_font)
#
##花呗支出
##income = final_income(10000,1500)["月收入"].tolist()
##expense=final_expense()["yzzc"].tolist()
##saving=[0 for i in range(120)]
##debt=[0 for i in range(120)]
##month=[]
##data=[]
##for i in range(120):
##    money=saving[i]+income[i]-debt[i]-expense[i]
##    if -money>15000:
##        print("第%i个月吃土\n-----" %(i+1))
##        break
##    else:
##        if money>=0:
##             saving[i+1]=income[i]-expense[i]+saving[i]-debt[i]
##             debt[i+1]=0 
##        else:
##            saving[i+1]=0
##            debt[i+1]=expense[i]-income[i]-(saving[i]-debt[i])
##    month.append(i+1)
##    data.append([income[i],expense[i],debt[i],saving[i+1],debt[i+1]])
##result_a=pd.DataFrame(data,columns=["月收入","月支出","本月要还花呗","本月剩余钱","欠债"],index=month)
##result_a.index.name="月份"
##print(result_a)
#
##模拟一万次
##income = final_income(10000,1500)["月收入"].tolist()
##expense=final_expense()["yzzc"].tolist()
##saving=[0 for i in range(120)]
##debt=[0 for i in range(120)]
##month=[]
##data=[]
##def case_a():
##    income = final_income(10000,1500)["月收入"].tolist()
##    expense=final_expense()["yzzc"].tolist()
##    saving=[0 for i in range(120)]
##    debt=[0 for i in range(120)]
##    month=[]
##    data=[]
##    for i in range(120):
##        money=saving[i]+income[i]-debt[i]-expense[i]
##        if -money>15000:
##            print("第%i个月吃土\n-----" %(i+1))
##            break
##        else:
##            if money>=0:
##                 saving[i+1]=income[i]-expense[i]+saving[i]-debt[i]
##                 debt[i+1]=0 
##            else:
##                saving[i+1]=0
##                debt[i+1]=expense[i]-income[i]-(saving[i]-debt[i])
##        month.append(i+1)
##        data.append([income[i],expense[i],debt[i],saving[i+1],debt[i+1]])
##    result_a=pd.DataFrame(data,columns=["月收入","月支出","本月要还花呗","本月剩余钱","欠债"],index=month)
##    result_a.index.name="月份"
##    return(result_a)
##case_a() 
##
##mouth_case_a=[]
##for i in range(100):
##    print("正在进行第%i次模拟" %(i+1))
##    income = final_income(10000,1500)["月收入"].tolist()
##    expense=final_expense()["yzzc"].tolist()
##    saving=[0 for i in range(120)]
##    debt=[0 for i in range(120)]
##    mouth_a=case_a().index.max()
##    mouth_case_a.append(mouth_a)
##result_a=pd.Series(mouth_case_a)
##plt.figure(figsize=(12,4))
##result_a.hist(bins=10)
##plt.title("不可分期0",fontproperties=my_font)
#
##花呗分期
#income = final_income(10000,1500)["月收入"].tolist()
#expense=final_expense()["yzzc"].tolist()
#saving=[0 for i in range(120)]
#debt=[0 for i in range(120)]
#month=[]
#data=[]
#def case_a():
#    income = final_income(10000,1500)["月收入"].tolist()
#    expense=final_expense()["yzzc"].tolist()
#    saving=[0 for i in range(120)]
#    debt=[0 for i in range(120)]
#    month=[]
#    data=[]
#    for i in range(120):
#        money=saving[i]+income[i]-debt[i]-expense[i]
#        if -money>15000:
#            print("第%i个月吃土\n-----" %(i+1))
#            break
#        else:
#            if money>=0:
#                 saving[i+1]=income[i]-expense[i]+saving[i]-debt[i]
#                 debt[i+1]=0 
#            else:
#                money_per=(abs(money)*(1+0.025))/3
#                saving[i+1]=0
#                debt[i+1]=debt[i+1]+money_per
#                debt[i+2]=debt[i+2]+money_per
#                debt[i+3]=debt[i+3]+money_per
# 
#        month.append(i+1)
#        data.append([income[i],expense[i],debt[i],saving[i+1],debt[i+1]])
#    result_a=pd.DataFrame(data,columns=["月收入","月支出","本月要还花呗","本月剩余钱","欠债"],index=month)
#    result_a.index.name="月份"
#    return(result_a)
#case_a() 