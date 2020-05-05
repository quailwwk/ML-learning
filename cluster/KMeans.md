 [sklearn.cluster.KMeans — scikit-learn 0.22.2 documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans.score) 

# 算法概述

K-Means算法是一种无监督分类算法，假设有无标签数据集：

![X= \left[ \begin{matrix} x^{(1)} \\ x^{(2)} \\ \vdots \\ x^{(m)} \\ \end{matrix} \right]](https://math.jianshu.com/math?formula=X%3D%20%5Cleft%5B%20%5Cbegin%7Bmatrix%7D%20x%5E%7B(1)%7D%20%5C%5C%20x%5E%7B(2)%7D%20%5C%5C%20%5Cvdots%20%5C%5C%20x%5E%7B(m)%7D%20%5C%5C%20%5Cend%7Bmatrix%7D%20%5Cright%5D)

该算法的任务是将数据集聚类成![k](https://math.jianshu.com/math?formula=k)个簇![C={C_{1},C_{2},...,C_{k}}](https://math.jianshu.com/math?formula=C%3D%7BC_%7B1%7D%2CC_%7B2%7D%2C...%2CC_%7Bk%7D%7D)，最小化损失函数为：

![E=\sum_{i=1}^{k}\sum_{x\in{C_{i}}}||x-\mu_{i}||^{2}](https://math.jianshu.com/math?formula=E%3D%5Csum_%7Bi%3D1%7D%5E%7Bk%7D%5Csum_%7Bx%5Cin%7BC_%7Bi%7D%7D%7D%7C%7Cx-%5Cmu_%7Bi%7D%7C%7C%5E%7B2%7D)

其中![\mu_{i}](https://math.jianshu.com/math?formula=%5Cmu_%7Bi%7D)为簇![C_{i}](https://math.jianshu.com/math?formula=C_%7Bi%7D)的中心点：

![\mu_{i}=\frac{1}{|C_{i}|}\sum_{x\in{C{i}}}x](https://math.jianshu.com/math?formula=%5Cmu_%7Bi%7D%3D%5Cfrac%7B1%7D%7B%7CC_%7Bi%7D%7C%7D%5Csum_%7Bx%5Cin%7BC%7Bi%7D%7D%7Dx)

![](https://images2018.cnblogs.com/blog/1366181/201804/1366181-20180402195806236-153083992.png)

# 划分方法

要找到以上问题的最优解需要遍历所有可能的簇划分，K-Means算法使用贪心策略求得一个近似解，具体步骤如下：

1. 在样本中随机选取![k](https://math.jianshu.com/math?formula=k)个样本点充当各个簇的初始中心点![\{\mu_{1},\mu_{2},...,\mu_{k}\}](https://math.jianshu.com/math?formula=%5C%7B%5Cmu_%7B1%7D%2C%5Cmu_%7B2%7D%2C...%2C%5Cmu_%7Bk%7D%5C%7D) 
2. 计算所有样本点与各个簇中心之间的距离![dist(x^{(i)},\mu_{j})](https://math.jianshu.com/math?formula=dist(x%5E%7B(i)%7D%2C%5Cmu_%7Bj%7D))，然后把样本点划入最近的簇中![x^{(i)}\in{\mu_{nearest}}](https://math.jianshu.com/math?formula=x%5E%7B(i)%7D%5Cin%7B%5Cmu_%7Bnearest%7D%7D) 
3. 根据簇中已有的样本点，重新计算簇中心
    ![\mu_{i}:=\frac{1}{|C_{i}|}\sum_{x\in{C{i}}}x](https://math.jianshu.com/math?formula=%5Cmu_%7Bi%7D%3A%3D%5Cfrac%7B1%7D%7B%7CC_%7Bi%7D%7C%7D%5Csum_%7Bx%5Cin%7BC%7Bi%7D%7D%7Dx) 
4. 重复2、3

## 缺陷

K-means算法得到的聚类结果严重依赖与初始簇中心的选择，如果初始簇中心选择不好，就会陷入局部最优解，如下图：

![img](https://upload-images.jianshu.io/upload_images/7312709-28808aed137a73f1.png?imageMogr2/auto-orient/strip|imageView2/2/w/511/format/webp)



![img](https://upload-images.jianshu.io/upload_images/7312709-e975851b71638ab0.png?imageMogr2/auto-orient/strip|imageView2/2/w/513/format/webp)





### 解决

#### 1.重复运行取平均

#### 2.KMeans++

​    初始的聚类中心之间的相互距离要尽可能的远。 

![img](https://images2018.cnblogs.com/blog/1366181/201804/1366181-20180402200209017-1976662980.png) 





##### 例子

 ![img](https://images2018.cnblogs.com/blog/1366181/201804/1366181-20180402200309449-1185158849.png) 

 假设6号点被选择为第一个初始聚类中心， 

 ![img](https://images2018.cnblogs.com/blog/1366181/201804/1366181-20180402200353467-1472082301.png) 

方法是随机产生出一个0~1之间的随机数，判断它属于哪个区间，那么该区间对应的序号就是被选择出来的第二个聚类中心了。

例如1号点的区间为[0,0.2)，2号点的区间为[0.2, 0.525)。

从上表可以直观的看到第二个初始聚类中心是1号，2号，3号，4号中的一个的概率为0.9。

而这4个点正好是离第一个初始聚类中心6号点较远的四个点。



# 初始点的选择

## K-Means++





# K的选择

值得一提的是关于聚类中心数目（K值）的选取，的确存在一种可行的方法，叫做Elbow Method：

通过绘制K-means代价函数与聚类数目K的关系图，选取直线拐点处的K值作为最佳的聚类中心数目。

上述方法中的拐点在实际情况中是很少出现的。

**比较提倡的做法还是从实际问题出发，人工指定比较合理的K值，通过多次随机初始化聚类中心选取比较满意的结果。**





## ISODATA算法

 K-means和K-means++的聚类中心数K是固定不变的。而ISODATA算法在运行过程中能够根据各个类别的实际情况进行两种操作来调整聚类中心数K：(1)**分裂操作**，对应着增加聚类中心数；(2)**合并操作**，对应着减少聚类中心数。 

### 基本思想

 **该算法能够在聚类过程中根据各个类所包含样本的实际情况动态调整聚类中心的数目。如果某个类中样本分散程度较大（通过方差进行衡量）并且样本数量较大，则对其进行分裂操作；如果某两个类别靠得比较近（通过聚类中心的距离衡量），则对它们进行合并操作。** 

### 参数

 **[1] 预期的聚类中心数目k0**：虽然在ISODATA运行过程中聚类中心数目是可变的，但还是需要由用户指定一个参考标准。事实上，该算法的聚类中心数目变动范围也由k0决定。具体地，最终输出的聚类中心数目范围是 [k0/2, 2k0]。 

  **[2] 每个类所要求的最少样本数目\*Nmin\***：用于判断当某个类别所包含样本分散程度较大时是否可以进行分裂操作。如果分裂后会导致某个子类别所包含样本数目小于***Nmin***，就不会对该类别进行分裂操作。 

 **[3] 最大方差\*Sigma\***：用于衡量某个类别中样本的分散程度。当样本的分散程度超过这个值时，则有可能进行分裂操作（注意同时需要满足**[2]**中所述的条件）。 

 **[4] 两个类别对应聚类中心之间所允许最小距离\*dmin\***：如果两个类别靠得非常近（即这两个类别对应聚类中心之间的距离非常小），则需要对这两个类别进行合并操作。是否进行合并的阈值就是由***dmin***决定。 

### 缺陷

 过由于它和其他两个方法相比需要额外指定较多的参数，并且某些参数同样很难准确指定出一个较合理的值，因此ISODATA算法在实际过程中并没有K-means++受欢迎。 



### 步骤

#### 主体部分

 ![图4](https://images2015.cnblogs.com/blog/1024143/201701/1024143-20170111025949447-680611657.png) 



#### 合并操作

 ![图5](https://images2015.cnblogs.com/blog/1024143/201701/1024143-20170111025951775-1194408309.png) 



#### 分裂操作

 ![图6](https://images2015.cnblogs.com/blog/1024143/201701/1024143-20170111025954494-895315300.png) 





# 带标签数据类内聚类

 K-means算法还可用于带标签的数据，在这种情况下，K-means会对每一个类别做单独的聚类。如某数据集可分为![C](https://math.jianshu.com/math?formula=C)个类别，那么K-means算法会将每一个类别看做是一个单独的数据集进行聚类操作。 



## 缺陷

 在不同类的数据有重叠的情况下，类内的聚类簇也会出现重叠现象，这是因为不同类之间的内部聚类是完全独立的，这样就造成类边界处的点极易被误分。 



##  **学习矢量量化**(LVQ,Learning Vector Quantization) 

### 基本思想

 核心思想是同类别的样本点会吸引簇中心，而不同类别的样本点会排斥簇中心 

### 算法

在每一个类别中都随机选取![R](https://math.jianshu.com/math?formula=R)个簇中心：![C_{1}^{[k]}](https://math.jianshu.com/math?formula=C_%7B1%7D%5E%7B%5Bk%5D%7D), ![C_{2}^{[k]}](https://math.jianshu.com/math?formula=C_%7B2%7D%5E%7B%5Bk%5D%7D), ![...](https://math.jianshu.com/math?formula=...), ![C_{R}^{[k]}](https://math.jianshu.com/math?formula=C_%7BR%7D%5E%7B%5Bk%5D%7D)，![k=1, 2, ..., K](https://math.jianshu.com/math?formula=k%3D1%2C%202%2C%20...%2C%20K) 

在所有数据中有放回地随机选取一个样本点![x_{i}](https://math.jianshu.com/math?formula=x_%7Bi%7D)![x_{i}](https://math.jianshu.com/math?formula=x_%7Bi%7D)![C_{r}^{[k]}](https://math.jianshu.com/math?formula=C_%7Br%7D%5E%7B%5Bk%5D%7D)![C_{r}^{[k]}](https://math.jianshu.com/math?formula=C_%7Br%7D%5E%7B%5Bk%5D%7D)

- 如果![C_{r}^{[k]}](https://math.jianshu.com/math?formula=C_%7Br%7D%5E%7B%5Bk%5D%7D)与![x_{i}](https://math.jianshu.com/math?formula=x_%7Bi%7D)同类，则将![C_{r}^{[k]}](https://math.jianshu.com/math?formula=C_%7Br%7D%5E%7B%5Bk%5D%7D)往![x_{i}](https://math.jianshu.com/math?formula=x_%7Bi%7D)的方向移动：![C_{r}^{[k]}:=C_{r}^{[k]}+{\alpha}(x_{i}-C_{r}^{[k]})](https://math.jianshu.com/math?formula=C_%7Br%7D%5E%7B%5Bk%5D%7D%3A%3DC_%7Br%7D%5E%7B%5Bk%5D%7D%2B%7B%5Calpha%7D(x_%7Bi%7D-C_%7Br%7D%5E%7B%5Bk%5D%7D)) 
- 如果![C_{r}^{[k]}](https://math.jianshu.com/math?formula=C_%7Br%7D%5E%7B%5Bk%5D%7D)与![x_{i}](https://math.jianshu.com/math?formula=x_%7Bi%7D)异类，则将![C_{r}^{[k]}](https://math.jianshu.com/math?formula=C_%7Br%7D%5E%7B%5Bk%5D%7D)往![x_{i}](https://math.jianshu.com/math?formula=x_%7Bi%7D)的反方向移动：![C_{r}^{[k]}:=C_{r}^{[k]}-{\alpha}(x_{i}-C_{r}^{[k]})](https://math.jianshu.com/math?formula=C_%7Br%7D%5E%7B%5Bk%5D%7D%3A%3DC_%7Br%7D%5E%7B%5Bk%5D%7D-%7B%5Calpha%7D(x_%7Bi%7D-C_%7Br%7D%5E%7B%5Bk%5D%7D)) 

重复2直到各簇中心不再变化或满足某种条件

 LVQ算法中的![\alpha](https://math.jianshu.com/math?formula=%5Calpha)为学习率，它会随着迭代次数而衰减至0 



在同样的数据上应用LVQ的聚类结果如下：

![img](https://upload-images.jianshu.io/upload_images/7312709-45dda58d1105c026.png?imageMogr2/auto-orient/strip|imageView2/2/w/686/format/webp)