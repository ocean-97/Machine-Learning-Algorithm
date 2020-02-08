## 第3章 线性回归

***

* 解决回归问题
* 思想简单，容易实现
* 许多非线性模型的基础
* 结果具备很好的可解释性
* 蕴含许多机器学习的重要思想

***

### 3.1 简单线性回归

​		（1）一个特征：简单线性回归 ；多个特征：多元线性回归。

​		（2）衡量损失的函数为损失函数，衡量拟合程度的函数为效用函数，两者统称为目标函数。

​		（3）通过给目标函数求取最佳参数得到模型，是近乎所有参数学习算法的思路：线性回归、多项式回归、逻辑回归、SVM、神经网络……，其中重要的数学基础为凸优化、最优化原理。

思路：

​		假设我们找到了最佳拟合的直线方程: $y=ax+b$, 则对于每一个点 $a^{(i)}$ , 

​		根据我们的直线方程，预测值为：$y^{(i)}=ax^{(i)}+b$ , 真值为：$y^{(i)}$。

目标：

​		使$\sum _{i=1}^n \left ( y^{(i)} -\widehat{y}^{(i)}\right )^2$ 尽可能小，$\because \widehat{y}^{(i)}= ax^{(i)}+b$ ，即目标函数为：
$$
\sum_{i=1}^n \left(y^{(i)}-ax^{(i)}-b\right)^2
$$
​		此时需找到合适的$a$和$b$, 使得目标函数的值最小，即：
$$
a = \frac{\sum_{i=1}^n \left(x^{(i)}-\overline{x} \right)\left(y^{(i)}-\overline{y} \right )}{\sum_{i=1}^n\left(x^{(i)}-\overline{x}\right)}\\b=\overline{y}-a\overline{x}
$$

#### 3.1.1 实现简单的线性回归

```python
import numpy as np
import matplotlib.pyplot as plt
x = np.array([1., 2., 3., 4., 5.])
y = np.array([1., 3., 2., 3., 5.])
x_mean = np.mean(x)
y_mean = np.mean(y)
# 求相关参数
num = 0.0
d = 0.0
for x_i, y_i in zip(x, y):
    num += (x_i - x_mean) * (y_i - y_mean)
    d += (x_i - x_mean) ** 2
a = num/d
b = y_mean - a * x_mean
y_hat = a * x + b
# 画图
plt.scatter(x, y)
plt.plot(x, y_hat, color='r')
plt.axis([0, 6, 0, 6])
plt.show()
# 测试
x_predict = 6
y_predict = a * x_predict + b
y_predict
```

#### 3.1.2 封装自己的简单线性回归算法

```Python
import numpy as np


class SimpleLinearRegression1:

    def __init__(self):
        """初始化Simple Linear Regression 模型"""
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        """根据训练数据集x_train,y_train训练Simple Linear Regression模型"""
        assert x_train.ndim == 1, \
            "Simple Linear Regressor can only solve single feature training data."
        assert len(x_train) == len(y_train), \
            "the size of x_train must be equal to the size of y_train"

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)
        
        # 传统计算方法
        num = 0.0
        d = 0.0
        for x, y in zip(x_train, y_train):
            num += (x - x_mean) * (y - y_mean)
            d += (x - x_mean) ** 2

        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_predict):
        """给定待预测数据集x_predict，返回表示x_predict的结果向量"""
        assert x_predict.ndim == 1, \
            "Simple Linear Regressor can only solve single feature training data."
        assert self.a_ is not None and self.b_ is not None, \
            "must fit before predict!"

        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        """给定单个待预测数据x，返回x的预测结果值"""
        return self.a_ * x_single + self.b_

    def __repr__(self):
        return "SimpleLinearRegression1()"    # 返回作为提示
```

```python 
# 调用上面代码
from playML.SimpleLinearRegression import SimpleLinearRegression1

reg1 = SimpleLinearRegression1()
reg1.fit(x, y)
reg1.predict(np.array([x_predict]))

print(reg1.a_, reg1.b_)    # 输出对应斜率与截距

y_hat1 = reg1.predict(x)
# 可视化
plt.scatter(x, y)
plt.plot(x, y_hat1, color='r')
plt.axis([0, 6, 0, 6])
plt.show()
```

#### 3.1.3 用向量化的方法提升性能

```
  # 用向量的方法对其改进
        num = (x_train - x_mean).dot(y_train - y_mean)
        d = (x_train - x_mean).dot(x_train - x_mean)
```



***

### 3.2 算法性能的衡量

```python 
"""
加载原始数据
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
# 波士顿房价数据
boston = datasets.load_boston()    # 创建实例
boston.keys()    # 字典的键
print(boston.DESCR)    # 可输出对数据集的介绍
boston.feature_names    # 每列数据的标记
x = boston.data[:,5]    # 只使用房间数量这个特征
y = boston.target    # 数据的标签
plt.scatter(x, y)    # 可视化数据
plt.show()
x = x[y < 50.0]    # 去掉最上方的干扰数据
y = y[y < 50.0]
# 使用简单线性回归
from playML.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, seed=666)

from playML.SimpleLinearRegression import SimpleLinearRegression
reg = SimpleLinearRegression()
reg.fit(x_train, y_train)

plt.scatter(x_train, y_train)    # 训练集以及模型可视化
plt.plot(x_train, reg.predict(x_train), color='r')
plt.show()

plt.scatter(x_train, y_train)    # 加入测试集对其可视化
plt.scatter(x_test, y_test, color="c")
plt.plot(x_train, reg.predict(x_train), color='r')
plt.show()
```

#### 3.2.1  误差评价的简单实现

```
mse_test = np.sum((y_predict - y_test)**2) / len(y_test)    #使用均方误差MSE

from math import sqrt    # 使用根均方误差RMSE
rmse_test = sqrt(mse_test)

mae_test = np.sum(np.absolute(y_predict - y_test))/len(y_test)    # 使用平均绝对误差MAE

1 - mean_squared_error(y_test, y_predict)/np.var(y_test)    # 使用R Square作为评价标准
```

#### 3.2.2 自行封装实现

```python
import numpy as np
from math import sqrt


def accuracy_score(y_true, y_predict):
    """计算y_true和y_predict之间的准确率"""
    assert len(y_true) == len(y_predict), \
        "the size of y_true must be equal to the size of y_predict"

    return np.sum(y_true == y_predict) / len(y_true)


def mean_squared_error(y_true, y_predict):
    """计算y_true和y_predict之间的MSE"""
    assert len(y_true) == len(y_predict), \
        "the size of y_true must be equal to the size of y_predict"

    return np.sum((y_true - y_predict)**2) / len(y_true)


def root_mean_squared_error(y_true, y_predict):
    """计算y_true和y_predict之间的RMSE"""

    return sqrt(mean_squared_error(y_true, y_predict))


def mean_absolute_error(y_true, y_predict):
    """计算y_true和y_predict之间的MAE"""

    return np.sum(np.absolute(y_true - y_predict)) / len(y_true)
  
  
def r2_score(y_true, y_predict):
    """计算y_true和y_predict之间的R Square"""

    return 1 - mean_squared_error(y_true, y_predict)/np.var(y_true)
```

```python
# 导入应用
from playML.metrics import mean_squared_error
from playML.metrics import root_mean_squared_error
from playML.metrics import mean_absolute_error
from playML.metrics import r2_score


mean_squared_error(y_test, y_predict)
root_mean_squared_error(y_test, y_predict)
mean_absolute_error(y_test, y_predict)
r2_score(y_test, y_predict)
```

#### 3.2.3 应用scikit-learn中的评价函数

```python
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

mean_squared_error(y_test, y_predict)
mean_absolute_error(y_test, y_predict)
r2_score(y_test, y_predict)
```

***

### 3.3 多元线性回归

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

boston = datasets.load_boston()

X = boston.data
y = boston.target

X = X[y < 50.0]
y = y[y < 50.0]

from playML.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, seed=666)
```



#### 3.3.1 封装自己的多元线性回归算法

封装

```python
import numpy as np
from .metrics import r2_score


class SimpleLinearRegression:

    def __init__(self):
        """初始化Simple Linear Regression模型"""
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        """根据训练数据集x_train, y_train训练Simple Linear Regression模型"""
        assert x_train.ndim == 1, \
            "Simple Linear Regressor can only solve single feature training data."
        assert len(x_train) == len(y_train), \
            "the size of x_train must be equal to the size of y_train"

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        self.a_ = (x_train - x_mean).dot(y_train - y_mean) / (x_train - x_mean).dot(x_train - x_mean)
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_predict):
        """给定待预测数据集x_predict，返回表示x_predict的结果向量"""
        assert x_predict.ndim == 1, \
            "Simple Linear Regressor can only solve single feature training data."
        assert self.a_ is not None and self.b_ is not None, \
            "must fit before predict!"

        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        """给定单个待预测数据x，返回x的预测结果值"""
        return self.a_ * x_single + self.b_

    def score(self, x_test, y_test):
        """根据测试数据集 x_test 和 y_test 确定当前模型的准确度"""

        y_predict = self.predict(x_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "SimpleLinearRegression()"
```

导入实现

```python
from playML.LinearRegression import LinearRegression

reg = LinearRegression()
reg.fit_normal(X_train, y_train)

print(reg.coef_, reg.intercept_)    # 输出系数以及截距
reg.score(X_test, y_test)     # 输出R Square评价值
```

#### 3.3.2 Scikit-Learn实现多元线性回归

```python
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

lin_reg.score(X_test, y_test)    # 输出评价值
```

#### 3.3.3 KNN线性回归

```python
# 对数据进行标准化
from sklearn.preprocessing import StandardScaler

standardScaler = StandardScaler()
standardScaler.fit(X_train, y_train)
X_train_standard = standardScaler.transform(X_train)
X_test_standard = standardScaler.transform(X_test)

# 引入KNN回归算法
from sklearn.neighbors import KNeighborsRegressor

knn_reg = KNeighborsRegressor()
knn_reg.fit(X_train_standard, y_train)
knn_reg.score(X_test_standard, y_test)

# 寻找最好的超参数
from sklearn.model_selection import GridSearchCV

param_grid = [
    {
        "weights": ["uniform"],
        "n_neighbors": [i for i in range(1, 11)]
    },
    {
        "weights": ["distance"],
        "n_neighbors": [i for i in range(1, 11)],
        "p": [i for i in range(1,6)]
    }
]

knn_reg = KNeighborsRegressor()
grid_search = GridSearchCV(knn_reg, param_grid, n_jobs=-1, verbose=1)
grid_search.fit(X_train_standard, y_train)

print(grid_search.best_params_)    # 输出最佳超参数
print(grid_search.best_score_)    # 输出最佳正确率，该值不能直接与上面的R Square比较
print(grid_search.best_estimator_.score(X_test_standard, y_test))    # 在最佳条件下测试集的正确率
```

