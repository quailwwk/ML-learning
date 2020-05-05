# 基本思想

用一个超平面把属于不同类别的样本分离开



# sklearn.svm

 cikit-learn中SVM的算法库分为两类，一类是分类的算法库，包括SVC， NuSVC，和LinearSVC 3个类。另一类是回归算法库，包括SVR， NuSVR，和LinearSVR 3个类。相关的类都包裹在sklearn.svm模块之中。 

## 数据要归一化

## 核函数选择---先尝试线性核

有一个经验法则是，**永远先从线性核函数开始尝试**（要记住，LinearSVC比SVC（kernel="linear"）快得多），特别是训练集非常大或特征非常多的时候， 使用线性核，效果已经很好，并且只需要选择惩罚系数C即可。 

如果训练集不太大或线性效果不好，你可以试试高斯RBF核， 主要需要对**惩罚系数C和核函数参数**γ进行艰苦的调参，通过多轮的交叉验证选择合适的惩罚系数C和核函数参数γ。 

```
gamma越大，维数越高，模型越复杂，越容易过拟合
gamma越小，越接近线性核

C越大，对异常变量引起损失的容忍度越低，模型越迁就于异常值，越容易过拟合
C越小，对异常变量的容忍度越高，模型会忽略异常值，但容易欠拟合。
```

 理论上高斯核不会比线性核差，但是这个理论却建立在要花费更多的时间来调参上。所以实际上能用线性核解决问题我们尽量使用线性核。 





# 优点

*1) 解决高维特征的分类问题和回归问题很有效,在特征维度大于样本数时依然有很好的效果。*

2) 仅仅使用一部分支持向量来做超平面的决策，无需依赖全部数据。

3) 有大量的核函数可以使用，从而可以很灵活的来解决各种非线性的分类回归问题。

4)样本量不是海量数据的时候，分类准确率高，泛化能力强。（软间隔）

# 缺点

1) 如果特征维度远远大于样本数，则SVM表现一般。

2) SVM在样本量非常大，核函数映射维度非常高时，计算量过大，不太适合使用。

3）非线性问题的核函数的选择没有通用标准，难以选择一个合适的核函数。

4）SVM对缺失数据敏感。



# 基本概念

## 函数间隔

![](https://quailwwk1.oss-cn-beijing.aliyuncs.com/typora截图/20200425150531.png)

![](https://quailwwk1.oss-cn-beijing.aliyuncs.com/typora截图/20200425150609.png)

若全部被正确分类，函数间隔应为正。函数间隔越大，说明分类效果越好。

## 几何间隔

![](https://quailwwk1.oss-cn-beijing.aliyuncs.com/typora截图/20200425150742.png)

保证w,b成比例变化时，间隔不变





# 模型步骤(线性可分)

## 最大(硬)间隔分类器

 $max\ \widetilde \gamma=\frac{\hat \gamma}{\parallel \omega \parallel}$ 

s.t.![](https://quailwwk1.oss-cn-beijing.aliyuncs.com/typora截图/20200425151526.png)

可令函数间隔为1（因为超平面本身和成比例变化的w\b无关，所以整个超平面空间和满足$\hat \gamma=1$的超平面子空间相等）

问题变为

![](https://quailwwk1.oss-cn-beijing.aliyuncs.com/typora截图/20200425151757.png)

## 优化问题转化

拉格朗日函数

![](https://quailwwk1.oss-cn-beijing.aliyuncs.com/typora截图/20200425151913.png)

![](https://quailwwk1.oss-cn-beijing.aliyuncs.com/typora截图/20200425151927.png)

易知

$$ \theta (\omega)= \left\{ \begin{aligned} \frac{1}{2} ||\omega||^2,\ if\ y_if(x_i)>=1\\ \infin,\ else \end{aligned} \right. $$

问题转化为

![](https://quailwwk1.oss-cn-beijing.aliyuncs.com/typora截图/20200425152639.png)

**对偶问题**

![](https://quailwwk1.oss-cn-beijing.aliyuncs.com/typora截图/20200425152706.png)

注：强对偶条件：slater+KKT

- **slater条件：保证鞍点存在**

  - 凸优化： 最小化问题，且目标函数和不等式约束函数均为凸函数，等式约束函数为仿射函数 

    => 局部最优=全局最优

  - 存在x，使得等式约束都成立，不等式约束都严格成立

- **KKT条件：确保鞍点是最优解**

  - 最优点必须是可行解 ![这里写图片描述](https://img-blog.csdn.net/20170316214502319?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvZmVpbG9uZ19jc2Ru/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast) 
  - 最优点处拉格朗日函数的梯度为0 ![这里写图片描述](https://img-blog.csdn.net/20170316214518179?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvZmVpbG9uZ19jc2Ru/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast) 
  - 不等式约束的拉格朗日乘子>=0 ，且与对应的不等式约束不同时不为零![这里写图片描述](https://img-blog.csdn.net/20170316214529538?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvZmVpbG9uZ19jc2Ru/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast) 





### [对偶问题的使用](https://segmentfault.com/a/1190000016328439)

并不一定要用拉格朗日对偶。要注意用拉格朗日对偶并没有改变最优解，而是改变了算法复杂度：
**在原问题下，求解算法的复杂度与样本维度（等于权值w的维度）有关；**
**而在对偶问题下，求解算法的复杂度与样本数量（等于拉格朗日算子a的数量）有关。**

因此，

1. 如果你是做**线性分类**，且**样本维度低于样本数量**的话，在**原问题**下求解就好了，Liblinear之类的线性SVM默认都是这样做的；
2. 如果你是做**非线性分类**，那就会涉及到**升维**（比如使用高斯核做核函数，其实是将样本升到无穷维），升维后的样本维度往往会远大于样本数量，此时显然在**对偶问题**下求解会更好。



## 对偶问题求解

### SMO算法

选择两个参数$\alpha_i$，$\alpha_j$，固定其他所有参数。然后求解最优化问题更新$\alpha_i$及$\alpha_j$，

先选取违背KKT条件程度最大的变量(更新后目标函数的增幅越大)，然后选择与其间隔最大的变量(两个参数差别越大，给目标函数带来的增幅就越大)

#### 优点：

仅优化两个参数的过程很高效。



## 结果

### 参数$\omega$

$\omega=\Sigma _{i=1}^{n} \alpha_iy_ix_i$

代入超平面方程有

![](https://quailwwk1.oss-cn-beijing.aliyuncs.com/typora截图/20200425180935.png)

 这里的形式的有趣之处在于，对于新点 *x*的预测，只需要计算它与训练数据点的内积即可（![img](https://img-blog.csdn.net/20131111163753093)表示向量内积），这一点至关重要，是之后使用 Kernel 进行非线性推广的基本前提。 

### 参数b

对任意支持向量$(x_s,y_s)$，都有$y_sf(x_s)=1$

将$\omega$代入即可解出一个b

b=使用所有支持向量求解b的平均值



# 非线性可分

将数据从原始空间映射到线性空间

$<x_i,x_j>\ =>\ <\phi(x_i),\phi(x_j)>$

若先给出具体映射，再求内积，计算内积时维度会很高



## 核函数

避开在高维空间中计算内积

 一般情况下，对非线性数据使用默认的高斯核函数会有比较好的效果，如果你不是SVM调参高手的话，建议使用高斯核来做数据分析。　 

### 充要条件

对任意数据集D，核矩阵为半正定



### 常用核函数

#### 多项式核

![](https://quailwwk1.oss-cn-beijing.aliyuncs.com/typora截图/20200425182135.png)

##### 优点：具体形式

##### 缺点：计算复杂,参数多

#### 高斯核RBF

![](https://quailwwk1.oss-cn-beijing.aliyuncs.com/typora截图/20200425182225.png)

##### 特点：

###### 无穷维

由exp(x)的泰勒展开知该核函数可以将数据映射到无穷维空间

###### 对特征数值敏感

要归一化

##### 调参

###### gamma:

gamma越大，维数越高，模型越复杂，越容易过拟合

gamma越小，越接近线性核



#### 线性核

![](https://quailwwk1.oss-cn-beijing.aliyuncs.com/typora截图/20200425182514.png)

##### 优点：计算简单

当样本数量和特征维度都很高时(尤其是特征数量很多)，用RBF会很慢(容易过拟合)，这时可以用线性核.





# 软间隔最大化

加入松弛变量，使得分类超平面受训练集中异常值的影响尽量小，防止过拟合·

![](https://quailwwk1.oss-cn-beijing.aliyuncs.com/typora截图/20200425185714.png)

## 调参

C越大，对异常变量引起损失的容忍度越低，模型越迁就于异常值，越容易过拟合

C越小，对异常变量的容忍度越高，模型会忽略异常值，但容易欠拟合。

## 松弛变量=损失函数

### hinge损失函数

$\xi_i=max(0,1-y_i(w^Tx_i+b))$ => ![image-20200425185934328](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20200425185934328.png)

### 指数损失

$\xi_i=exp(-y_i(w^Tx_i+b))$ 

### 对率损失

$\xi_i=log(1+exp(-\omega ^Tx_i+b))$



## 求解

与硬间隔时类似，区别是参数$\alpha_i>=0$的同时还有$\alpha_i<=C$





# 支持向量回归SVR

## 基本思想

 可容忍 f(x) 与 y 之间的差别绝对值最多为 ε ，仅当两者差值大于 ε 才计算损失。相当于以 f(x) 为中心，构建了一个宽度为 2ε 的间隔带，若训练样本落入此隔离带，则认为预测正确。 

 ![这里写图片描述](https://img-blog.csdn.net/20170904153230435?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbmlhb2xpYW5qaXVsaW4=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast) 

![](https://quailwwk1.oss-cn-beijing.aliyuncs.com/typora截图/20200425190818.png)

## 松弛变量--两个！

 考虑到**间隔带两侧的松弛程度可有所不同**，则有两个变量 $\xi_i,\xi_j$

![](https://quailwwk1.oss-cn-beijing.aliyuncs.com/typora截图/20200425190940.png)





## 稀疏性

KKT条件：![](https://quailwwk1.oss-cn-beijing.aliyuncs.com/typora截图/20200425191722.png)

若：

![](https://quailwwk1.oss-cn-beijing.aliyuncs.com/typora截图/20200425191905.png)

从而上式$<\epsilon+\xi_i$。

由KKT条件知，![](https://quailwwk1.oss-cn-beijing.aliyuncs.com/typora截图/20200425192139.png)

又由![](https://quailwwk1.oss-cn-beijing.aliyuncs.com/typora截图/20200425192204.png)

知，**$\omega$不受误差范围内的点的影响**





