# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 16:33:25 2019

@author: 92156
"""

def hcsl(b):
    q=70;e=70;r=70 # 第一、三、四组70辆
    n=int(b/70)#求满编组 6
    x=b%(n*70)#取余数
    if (x<50):#余数小于20
        q=70-(50-x); #将第一组的调动到第二组
        c=50-x;#需要调剂的数量
        if (c<20): #
             #第一组减去调动的
            x=x+c;#第二组获得第一组的车
        else:
            q=q-20;
            x=x+20;
            c=50-x;
            if (c<20):
                e=e-c; #第三组减去车
                x=x+c;
            else:
                e=e-20;
                x=x+20;
                c=50-x;
                if(c<20):
                    r=r-c;#第四组减去的车
                    x=x+c;
                else:
                    print("错误");
    if x>=50:# 如果余数x大于50
        hcz=pd.read_excel("sss.xlsx")
        q=int(q);x=int(x);e=int(e);r=int(r)
        hcz[0]=q;hcz[1]=x;hcz[2]=e;hcz[3]=r;
        print(hcz)
        hcz=np.array(hcz)
        return hcz,n
    
import pandas as pd 
import numpy as np 
z= list([ 0 ] * 1000)
zl= 0
cc1=pd.read_excel("18.xlsx")
cc2=pd.read_excel("181.xlsx")
cc4=int(cc2.sum(axis=1)/2)  #A2早
cc3=int(cc1.sum(axis=1)/2)  #A1早
cc5=int(cc3)   #A1晚
cc6=int(cc4)  #A2晚  
hcz,n=hcsl(int (cc1["黑"]))
print (n)
hcz=hcz.reshape(4,1)
R=int(cc1["黄"]+cc1["红"])
S=int(cc1["灰"]+cc1["银"])
#for x in range(0,int(hcz[0])):
#    z[zl]=0;
#    zl=zl+1
#    cc1["黑"]=int(cc1["黑"])-int(hcz[0]);  
#    cc4=cc4-int(hcz[0]); #错误 迭代出错 ，迭代了70次
#    if  int(cc1["蓝"])!=0:
#        if zl%2==0:
#            z[int(zl)]=2;
#            zl=zl+1
#            cc3=cc3-1
#            cc1["蓝"]=cc1["蓝"]-1
#            if cc3==0:
#                break
#            else:
#                z[int(zl)]=1;
#                zl=zl+1
#                cc3=cc3-1
#                cc1["白"]=cc1["白"]-1
#                if cc3==0:
#                    break
                

# 白天A1

            
a1hcsl=0
if int(cc2["黑"])>=cc4:
    a1hcsl=int(hcz[1])-cc4
else:
    a1hcsl=int(hcz[1])-int(cc2["黑"])
    
#    金车 R车(红黄)
while int(cc1["金"])!=0 and R!=0 and cc3!=a1hcsl:
    if zl%2==0:
        z[int(zl)]=5;
        zl=zl+1
        cc3=cc3-1
        R=R-1
        if cc3==a1hcsl:
            break
        else:
             z[int(zl)]=4;
             zl=zl+1
             cc3=cc3-1
             cc1["金"]=int(cc1["金"])-1
             if cc3==0:
                 break
    else:
         z[int(zl)]=4;
         zl=zl+1
         cc3=cc3-1
         cc1["金"]=int(cc1["金"])-1
         if cc3==0:
             break
         else:
             z[int(zl)]=5;
             zl=zl+1
             cc3=cc3-1
             R=R-1
             if cc3==a1hcsl:
                 break
             
           
if int(cc1["金"])==0:    # 循环7次
    #R S车
    while R!=0 and S!=0 and cc3>a1hcsl:
        if  zl%2==0:
            z[int(zl)]=5;
            zl=zl+1
            cc3=cc3-1
            R=R-1
            if cc3==a1hcsl:
                break
            else:
                z[int(zl)]=6;
                zl=zl+1
                cc3=cc3-1
                S=S-1
                if cc3==a1hcsl:
                    break
                print("cc1")
        else:
            z[int(zl)]=6;
            zl=zl+1
            cc3=cc3-1
            S=S-1
            if cc3==a1hcsl:
                    break
            else:
                z[int(zl)]=5;
                zl=zl+1
                cc3=cc3-1
                R=R-1
                if cc3==a1hcsl:
                    break
                
    if S==0:
        
    #R z棕
        while R!=0 and cc3>a1hcsl and int(cc1["棕"]):
            if zl%2==0:
                z[int(zl)]=5;
                zl=zl+1
                cc3=cc3-1
                R=R-1
                if cc3==a1hcsl:
                    break
                else:
                    z[int(zl)]=3;
                    zl=zl+1
                    cc3=cc3-1
                    S=S-1
                    if cc3==a1hcsl:
                        break
            else:
                 z[int(zl)]=3;
                 zl=zl+1
                 cc3=cc3-1
                 cc1["棕"]=int(cc1["棕"])-1
                 if cc3==a1hcsl:
                     break
                 else:
                     z[int(zl)]=5;
                     zl=zl+1
                     cc3=cc3-1
                     R=R-1
                     if cc3==a1hcsl:
                         break
elif R==0:
    #金和S
    while int(cc1["金"])!=0 and cc3>a1hcsl and S!=0:
        if zl%2==0:
            z[int(zl)]=4;
            zl=zl+1
            cc3=cc3-1
            cc1["金"]=int(cc1["金"])-1
            if cc3==a1hcsl:
                break
            else:
                z[int(zl)]=6;
                zl=zl+1
                cc3=cc3-1
                S=S-1
                if cc3==a1hcsl:
                    break
        else:
            z[int(zl)]=6;
            zl=zl+1
            cc3=cc3-1
            S=S-1
            if cc3==a1hcsl:
                break
            else:
                z[int(zl)]=4;
                zl=zl+1
                cc3=cc3-1
                cc1["金"]=int(cc1["金"])-1
                if cc3==a1hcsl:
                    break
                # 金棕
    if S==0:
        while int(cc1["金"])!=0 and int(cc1["棕"]!=0 and cc3>a1hcsl):
             if zl%2==0:
                 z[int(zl)]=4;
                 zl=zl+1;
                 cc3=cc3-1;
                 cc1["金"]=int(cc1["金"])-1;
                 if cc3==a1hcsl:
                     break
                 else:
                     z[int(zl)]=3;
                     zl=zl+1;
                     cc3=cc3-1;
                     cc1["棕"]=int(cc1["棕"])-1;
                     if cc3==a1hcsl:
                         break
             else:
                 z[int(zl)]=3;
                 zl=zl+1;
                 cc3=cc3-1;
                 cc1["棕"]=int(cc1["棕"])-1;
                 if cc3==a1hcsl:
                     break
                 else:
                     z[int(zl)]=4;
                     zl=zl+1;
                     cc3=cc3-1;
                     cc1["金"]=int(cc1["金"])-1;
                     if cc3==a1hcsl:
                         break
                     
        # S车   
while S!=0 and cc3!=a1hcsl:
    cc3=cc3-1
    S=S-1
    z[zl]=6;
    zl=zl+1
while int(cc1["白"])!=0 and int(cc1["棕"])!=0 and cc3!=a1hcsl:
     z[int(zl)]=1;
     zl=zl+1;
     cc3=cc3-1;
     cc1["白"]=int(cc1["白"])-1;
     if cc3==a1hcsl:                  
         break
     else:
         z[int(zl)]=3;
         zl=zl+1;
         cc3=cc3-1;
         cc1["棕"]=int(cc1["棕"])-1;
         if cc3==a1hcsl:
             break
if int(cc1["白"])==0:
    while cc3!=a1hcsl and int(cc1["白"])!=0:
        cc3=cc3-1
        cc1["棕"]=int(cc1["棕"])-1;
        z[zl]=3
        zl=zl+1
elif int(cc1["棕"])==0:
    while cc3!=a1hcsl and int(cc1["白"])!=0:
        cc1["白"]=int(cc1["白"])-1
        cc3=cc3-1
        z[zl]=1
        zl=zl+1
        
        
    
                    
                         
                    
                 
                
        

    
    