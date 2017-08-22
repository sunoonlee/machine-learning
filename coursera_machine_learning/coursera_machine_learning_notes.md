Coursera Machine Learning 课程笔记

- 课程页面: https://www.coursera.org/learn/machine-learning
- 网上一份很详细的笔记 → http://www.holehouse.org/mlclass/

## 目录
<!-- toc -->

- [w1 intro + 单变量线性回归 + 线代回顾](#w1-intro--%E5%8D%95%E5%8F%98%E9%87%8F%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92--%E7%BA%BF%E4%BB%A3%E5%9B%9E%E9%A1%BE)
- [w2 多变量线性回归 + octave](#w2-%E5%A4%9A%E5%8F%98%E9%87%8F%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92--octave)
  * [数学记法](#%E6%95%B0%E5%AD%A6%E8%AE%B0%E6%B3%95)
  * [Gradient descent in practice](#gradient-descent-in-practice)
  * [normal equation](#normal-equation)
  * [作业: 线性回归的梯度下降法](#%E4%BD%9C%E4%B8%9A-%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92%E7%9A%84%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E6%B3%95)
- [w3 Logistic regression + regularization](#w3-logistic-regression--regularization)
- [w4 neural networks: representation](#w4-neural-networks-representation)
- [w5 neural networks: learning](#w5-neural-networks-learning)
- [w6 advice for applying machine learning](#w6-advice-for-applying-machine-learning)
- [w7 SVM](#w7-svm)
  * [Large Margin Classification](#large-margin-classification)
  * [Kernels](#kernels)
  * [SVMs in Practice](#svms-in-practice)
  * [Assignment (ex6)](#assignment-ex6)
- [w8 Clustering](#w8-clustering)
  * [K-means: 最常用的聚类算法](#k-means-%E6%9C%80%E5%B8%B8%E7%94%A8%E7%9A%84%E8%81%9A%E7%B1%BB%E7%AE%97%E6%B3%95)
- [w8 Dimensionality reduction](#w8-dimensionality-reduction)
  * [motivation of dimensionality reduction](#motivation-of-dimensionality-reduction)
  * [PCA](#pca)
  * [advice for applying PCA](#advice-for-applying-pca)
- [w8 Assignment (ex7)](#w8-assignment-ex7)
  * [K-means](#k-means)
  * [PCA](#pca-1)
- [w9 Anomaly detection](#w9-anomaly-detection)
  * [Density estimation](#density-estimation)
  * [Building an anomoly detection system](#building-an-anomoly-detection-system)
  * [Multivariate Gaussian Distribution](#multivariate-gaussian-distribution)
- [w9 Recommender system](#w9-recommender-system)
  * [引子: content-based recommendations](#%E5%BC%95%E5%AD%90-content-based-recommendations)
  * [Collaborative filtering](#collaborative-filtering)
  * [Low rank matrix factorization](#low-rank-matrix-factorization)
  * [mean normalization](#mean-normalization)
- [w10 Large Scale Machine Learning](#w10-large-scale-machine-learning)
  * [SGD 和 mini-batch](#sgd-%E5%92%8C-mini-batch)
  * [online learning](#online-learning)
  * [map reduce and data parellelism](#map-reduce-and-data-parellelism)
- [w11 application example: photo OCR](#w11-application-example-photo-ocr)
  * [sliding window](#sliding-window)
  * [artificial data synthesis](#artificial-data-synthesis)
  * [ceiling analysis](#ceiling-analysis)

<!-- tocstop -->

## w1 intro + 单变量线性回归 + 线代回顾
- what is machine learning
  - Arthur Samuel: "the field of study that gives computers the ability to learn without being explicitly programmed."
    - 不依赖显式编程的学习
    - an older, informal definition
  - Tom Mitchell: "A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E."
    - 经验E, 任务T, 表现P
- octave: a good tool for learning and prototyping
- 梯度下降: 编程时注意不同参数需同时更新


## w2 多变量线性回归 + octave

这章学习遇到的难点是编程实现时的 vectorization

### 数学记法

- 令 x_0 = 1, 把 x 凑成 n+1 维向量. 这样数学表达更简洁.
- m, 上标 i: 训练样本总数及序号
- n, 下标 j: feature 总数及序号
- **dimensional analysis is your friend**

### Gradient descent in practice

- **feature scaling**: make sure features are on a similar scale.
  - get every feature into approximately a [-1, 1] range. 这样梯度下降收敛得更快。
- 更严格的: **mean normalization**. 使各 feature 均值都接近 0.
- 选择合适的 learning rate: 按3倍或10倍的比例尝试不同的值

### normal equation

- computing parameters analytically
- closed form solution： `theta = pinv(X’*X)*X*y`
- normal equation vs. gradient descent
  - normal equation: O(n^3). gradient descent: O(kn^2)
  - 当 n 很大时 (如大于 10^4), normal equation 会很慢, 建议用梯度下降
- normal equation noninvertibility
  - 求逆用 pinv 代替 inv
  - X^TX 不可逆的可能原因
    - redundant features (linearly dependent)
    - too many features (eg. m <= n)  → delete some features, or use regularization

### 作业: 线性回归的梯度下降法

- 梯度计算
  - 方法1：循环
  - 方法2：简洁的向量化解法
    - `delta = alpha / m * X' * (X*theta - y);`
    - theta = theta - delta;
    - 计算 logistic 回归梯度只需把 `X*theta` 换成 `g(X*theta)`


## w3 Logistic regression + regularization
- logistic regression
  - cost function 与参数更新公式

![](https://d2mxuefqeaa7sj.cloudfront.net/s_C71F128EC1AD0AA0CC664FF1B2FF7C9CC5A870050A4FC3E2D26DBB5320946F3D_1501909230031_image.png)


![](https://d2mxuefqeaa7sj.cloudfront.net/s_C71F128EC1AD0AA0CC664FF1B2FF7C9CC5A870050A4FC3E2D26DBB5320946F3D_1501909289680_image.png)

- 向量化:

![](https://d2mxuefqeaa7sj.cloudfront.net/s_C71F128EC1AD0AA0CC664FF1B2FF7C9CC5A870050A4FC3E2D26DBB5320946F3D_1501909308595_image.png)


![](https://d2mxuefqeaa7sj.cloudfront.net/s_C71F128EC1AD0AA0CC664FF1B2FF7C9CC5A870050A4FC3E2D26DBB5320946F3D_1501909314486_image.png)

- 较为复杂的优化算法: conjugate gradient, BFGS, L-BFGS. 
  - 可能的优点: 可自动选择alpha，收敛速度快，等
- 多分类问题: one-vs-all 方法
- 过拟合的解决: 减少特征数量; 正则化
  - 正则化 works well when we have a lot of **slightly useful** features
  - 在线性回归的 normal equation 方法里加入正则化, 可以解决矩阵不可逆的问题
  - 正则化不应涉及 bias 项


## w4 neural networks: representation
- notation
  - $a_i^{(j)}$ : 第j层第i单元的 “activation”
  - $\Theta^{(j)}$ : 从第j层映射到第j+1层的权重矩阵
  - 例: 3个输入节点, 3个隐层节点的单隐层神经网络的数学表达

![](https://d2mxuefqeaa7sj.cloudfront.net/s_C71F128EC1AD0AA0CC664FF1B2FF7C9CC5A870050A4FC3E2D26DBB5320946F3D_1501915324317_image.png)

- 把各层激活前的值向量记做 $z^{(j)}$, 则 $z^{(j)} = \Theta^{(j-1)} a^{(j-1)}$ , $a^{(j)} = g(z^{(j)})$
- octave tips
  - [匿名函数](https://www.gnu.org/software/octave/doc/interpreter/Anonymous-Functions.html#Anonymous-Functions)
  - logical arrays: `y == c`
  - 由行向量得到列向量: `vec = vec(:)`
- ex3 的材料里有一些对向量化的解释

![](https://d2mxuefqeaa7sj.cloudfront.net/s_C71F128EC1AD0AA0CC664FF1B2FF7C9CC5A870050A4FC3E2D26DBB5320946F3D_1501915874931_image.png)


![](https://d2mxuefqeaa7sj.cloudfront.net/s_C71F128EC1AD0AA0CC664FF1B2FF7C9CC5A870050A4FC3E2D26DBB5320946F3D_1501915929077_image.png)

## w5 neural networks: learning
- 神经网络的 cost function
  - 基本就是在逻辑回归的基础上增加一层对不同输出unit/分类的求和.
  - notation: L - 总层数, $s_l$ - 第 l 层的 unit 数, 不含 bias

![](https://d2mxuefqeaa7sj.cloudfront.net/s_C71F128EC1AD0AA0CC664FF1B2FF7C9CC5A870050A4FC3E2D26DBB5320946F3D_1501916235205_image.png)

- 梯度的计算: 反向传播
  - 先前向传播 计算各层激活值 $a^{(l)}$
  - 反向计算各层的偏差值 $\delta$ 

![](https://d2mxuefqeaa7sj.cloudfront.net/s_C71F128EC1AD0AA0CC664FF1B2FF7C9CC5A870050A4FC3E2D26DBB5320946F3D_1501916822641_image.png)

- 利用 $\delta$ 更新 $\Delta$ 矩阵 

![](https://d2mxuefqeaa7sj.cloudfront.net/s_C71F128EC1AD0AA0CC664FF1B2FF7C9CC5A870050A4FC3E2D26DBB5320946F3D_1501916865910_image.png)

- 以上三步对所有样本循环
  - 最后根据 $\Delta$ 矩阵得到梯度 D 矩阵

![](https://d2mxuefqeaa7sj.cloudfront.net/s_C71F128EC1AD0AA0CC664FF1B2FF7C9CC5A870050A4FC3E2D26DBB5320946F3D_1501916961555_image.png)

- 反向传播实施中的一些要点
  - unrolling parameters
    - 具体计算时，需要把 Theta 矩阵 unroll 成一阶向量，便于传给优化函数
  - gradient checking
    - 可以用数值方法计算梯度，检验反向传播算法对否：取一个 $\epsilon$, 则梯度约等于 $\frac{J(\theta+\epsilon) - J(\theta-\epsilon)}{2\epsilon}$ 。检验完之后，在训练时要记得禁用这个检验计算。
  - 随机初始化
    - 逻辑回归可以把参数都初始化为0, 但神经网络不行, 需要用随机初始化来打破对称性.
  - output 的形式: 如果 y 取值是 {1, 2, 3} 这样的, 在神经网络中的输出应该处理为 one-hot 向量 [1, 0, 0], [0, 1, 0], [0, 0, 1] , 具体可用 octave 的 logical arrays.


## w6 advice for applying machine learning
- 不要想一开始就搭建复杂的算法，可以先从一个简单的原型开始，a quick and dirty implementation。再复杂的算法, 搭建原型都不建议超过一天。
- 进一步，尝试增加样本数/引入更多特征/引入正则化 等等，**画出 J_train 和 J_val 随这些因素 (样本数, 特征数, lambda, …) 的变化曲线**, 判定模型是否偏向 high-bias 或 high-variance, 由此决定下一步行动。**做决定不能依赖直觉，要有依据**。
- bias-variance tradeoff
  - debugging 武器: **学习曲线**
  - fix high-bias: 增加新特征，增加多项式特征，减小 lambda
  - fix high-variance: 增大样本数量，精简特征，增大 lambda

![](https://d2mxuefqeaa7sj.cloudfront.net/s_C71F128EC1AD0AA0CC664FF1B2FF7C9CC5A870050A4FC3E2D26DBB5320946F3D_1501918418504_image.png)

![](https://d2mxuefqeaa7sj.cloudfront.net/s_C71F128EC1AD0AA0CC664FF1B2FF7C9CC5A870050A4FC3E2D26DBB5320946F3D_1501918427344_image.png)

- 另一有用武器: error analysis
  - 手动检查出错的样本, 看能否发现规律, 再针对性地改进.
- 借助合适的指标来衡量算法表现
- handling skewed data
  - 不同分类的数量比例悬殊时, 不宜用 accuracy 等单一指标, 可以结合 precision/recall, 或者用 f1 score. 
  - 具体情境下对 precision/recall 的偏好可能不同, 可通过调整 threshold 等方法来实现
- 参考: cs229幻灯 [ML-advice](http://cs229.stanford.edu/materials/ML-advice.pdf)


## w7 SVM

- 相关笔记
  - LFD +ml_LearningFromData2_lim: Lec14-SVM 
  - 统计学习方法 +ml_LiHangBook_lim: Ch7-SVM 

### Large Margin Classification

- [v] optimization objective
  - 这一节从逻辑回归引入 SVM, 主要对比了它们的单点 cost 函数. 这个角度有点特别, 不像其他教程里的思路: 什么是间隔 → 硬间隔 → 软间隔.
  - 假设函数: 逻辑回归为连续函数, SVM 为符号函数
  - 比较 cost function
    - 逻辑回归
      - 第一项源于交叉熵, 第二项是 weight decay 正则化

![](https://d2mxuefqeaa7sj.cloudfront.net/s_C71F128EC1AD0AA0CC664FF1B2FF7C9CC5A870050A4FC3E2D26DBB5320946F3D_1501137863250_image.png)

    - SVM (软间隔)
      - 第一项里的 cost_0, cost_1 就是 “合页损失函数”, 也是软间隔引入的”松弛变量”, 是对误分类的惩罚
      - 第二项对应的是间隔最大化
      - 不过这里没有列出 SVM 需要满足的不等式约束条件

![](https://d2mxuefqeaa7sj.cloudfront.net/s_C71F128EC1AD0AA0CC664FF1B2FF7C9CC5A870050A4FC3E2D26DBB5320946F3D_1501137885979_image.png)

- 比较单个数据点的 cost
    - 逻辑回归: 正/负例点分别为 $\log h(x^{(i)})$, $1-\log h(x^{(i)})$. 其中 $h(x) = \sigma(\theta^Tx)$
    - SVM: 把上面两个曲线替换为折线, 也就是“合页损失函数”. 计算代价会变小.

![](https://d2mxuefqeaa7sj.cloudfront.net/s_C71F128EC1AD0AA0CC664FF1B2FF7C9CC5A870050A4FC3E2D26DBB5320946F3D_1495862185614_image.png)

- [v] large margin intuition
  - 上面提到 SVM 正/负例点的 cost 与横轴交点分别为 -1/1. 这代表对分类准确性要求较严格(相比感知机), 对应 margin.

### Kernels

- [v] Kernels I
  - non-linear decision boundary
    - 如何获得非线性特征? 一种方法是引入高次项，但会造成参数爆炸
    - 可以利用与 landmarks 的距离/相似性 定义 features. 不同的距离/相似性定义对应不同的 kernel functions.
    - 可以把每个样本选为一个 landmark
    - 高斯核 $K(x^{(i)},y^{(i)}) = \exp (-\frac{||x^{(i)} - x^{(j)}||^2}{2\sigma^2})$
- [v] Kernels II
  - 核方法也可应用于其他算法，但此时无法使用 SVM 中所用的计算优化 tricks
  - SVM 参数
    - C: 相当于 $1/\lambda$
    - 高斯核的 $\sigma ^2$: 该参数越大，越趋向 high-bias

### SVMs in Practice
[v] using a svm

- SVM 实施要点
  - SVM 优化的求解较为复杂，一般用现成的经过优化的工具，而不是自己写代码去实现。(矩阵求逆等也类似)
    - liblinear, libsvm, ...
  - 调包需要指定:
    - 参数 C
    - 选 kernel：no kernel / Gaussian Kernel / ...
      - 当 n 很大、m 很小时，为防止过拟合，可以简单地选 no kernel, 也称 linear kernel
  - **!! 使用高斯核之前务必做 feature scaling**
  - kernel function  不能随便选择，需要满足 Mercer's Theorem. 一般 linear 和 Gaussian 两种够用了
  - 多分类: 很多 SVM 模块内置了对多分类的支持. 如果不支持, 可以用 one-vs.-all 方法.
- LR, SVM 选哪个?
  - 三种情况 (n为特征数量, m为数据量)
    - 特征多数据少, n >= m：用 LR 或 线性 SVM
    - n小 (1-1000), m不太大(<10000): 高斯核SVM
    - n小 m大: 这时候若继续用高斯核SVM, 速度会比较慢. 可以人为增加特征数, 然后用 LR 或线性SVM
  - LR 和 线性SVM 效果差不多
  - SVM vs. 神经网络
    - 有些问题上, 经过优化的 SVM 常比 NN 快
    - SVM 求解的是凸优化问题, 不容易碰到局部最小值
- 选哪种算法没那么重要，更重要的是：
  - 数据
  - 算法实施技巧：比如特征的选择，error analysis，debugging 等

### Assignment (ex6)

- Part1: SVM intuition
  - Most SVM software packages automatically add the extra feature x0 = 1 for you and automatically take care of learning the intercept term $\theta_0$
  - octave 计算预测错误率：`mean(double(predictions ~= yval))`
- Part2: Spam classification
  - 预处理
    - **normalize 一些特殊 entities, 如 URL, email, 数字, 金额. (用正则表达式)**
    - 其他: 大小写, stemming, 等.
    - tokenize
      - 低频词会造成过拟合. 实际中常用的词表大小是 1W-5W.
      - 字符串比较 `strcmp()`
## w8 Clustering

### K-means: 最常用的聚类算法

- 算法与目标函数
  - iterative algorithm. repeat: { 1. cluster assignment, 2. move centroid }
  - 目标函数是 $J = \frac{1}{m} \sum_{i=1}^m || x^{(i)} - \mu_{c^{(i)}}||^2$. (可以利用 J 随 iterations 变化的曲线来检查收敛性)
  - 模型参数包括: 各样本点被分配的 cluster 编号 $c^{(i)}$ ; 各 cluster 中心 $\mu_k$. 
  - 第一步是固定 $c^{(i)}$ 优化 $\mu_k$, 第二步是固定 $\mu_k$ 优化 $c^{(i)}$.
- 随机初始化
  - 一般随机选 K 个点作为初始的中心点
  - 算法结果受初始化值的影响, 可能落入局部最小值. 可以多试几次 (常见 50-1000次), 取最终损失函数最小的. → 这种做法在 K 较小时很管用, K 很大时作用不明显.
- 选择超参数 K
  - 一种方法是 Elbow method, 找 J-K 曲线的转折点. 但很可能曲线并没有明显的 Elbow.
  - K-means 经常作为其他任务的前置任务, 这时可以根据后续任务的表现来选 K.

![](https://d2mxuefqeaa7sj.cloudfront.net/s_C71F128EC1AD0AA0CC664FF1B2FF7C9CC5A870050A4FC3E2D26DBB5320946F3D_1501211283519_image.png)

## w8 Dimensionality reduction

### motivation of dimensionality reduction

- data compression: 可以减少数据存储需要的内存/硬盘空间; 可以加快机器学习速度
- data visualization: 高维数据不方便可视化, 可以降到2D/3D.

### PCA

- PCA 的目标: 找到一个低维空间, 使得 projection error 最小
- **data preprocessing**
  - **mean normalization: 必做**
  - **feature scaling: 选做**
    - 减去均值后 除以最值或标准差
- 算法步骤
  - 计算协方差矩阵 $\mathrm{Sigma} = \frac{1}{m} \sum_{i=1}^m (x^{(i)}) (x^{(i)})^T$
    - 若 X 是行向量形式的输入矩阵, 则 `Sigma = 1/m * X``'` `* X;`
  - `[U, S, V] = svd(Sigma);`
  - `Ureduce = U(:, 1:k);`
  - `z = Ureduce``'` `* x;`
- 数据”解压缩”: x_approx = U_reduce * z
- 如何选择 k (number of pricipal components)

![](https://d2mxuefqeaa7sj.cloudfront.net/s_C71F128EC1AD0AA0CC664FF1B2FF7C9CC5A870050A4FC3E2D26DBB5320946F3D_1501222858150_image.png)

- 控制这个比例: (average squared projection error) / (total variance in the data)
  - 常见限值是 1%~5%. 有时也会用到 10%-15%.
  - 可以利用 SVD 分解得到的对角矩阵 S. 选择满足下式的最小的 k: $\frac{\sum_{i=1}^k S_{ii}} {\sum_{i=1}^m S_{ii}} \geq 0.99$

### advice for applying PCA

- PCA 可以用来加速有监督学习
  - 从高维特征空间映射到低维. 有些问题中, 可以利用 PCA 将特征数量减少至 1/5 或 1/10 而几乎不影响精度.
  - PCA 的训练只能利用训练集.
- 两类应用场景
  - compression: 又分两类目标: 节省存储空间; 加快学习速度
    - 这种场景下 k 值根据 “percentage of retained variance” 来选
  - visualization
    - 这种场景下 一般选 k=2, 3, 便于绘图
- 不建议把 PCA 当做防止过拟合的措施
  - 正则化是更好的选择
  - 原因: PCA 的”压缩”过程只利用了 x, 没有利用 label, 因而可能丢失有用的信息. 而正则化过程可以利用到 label 值.
- 不要在有监督学习项目中轻易 (理所当然地) 引入 PCA
  - 尽量用原始数据来实施有监督学习算法
  - 只在以下三种情况下考虑引入 PCA:
    - 特征太多导致训练速度过慢
    - 特征太多导致占用内存/磁盘空间太大
    - 需要把数据降到2-3维以进行可视化


## w8 Assignment (ex7)
- 有趣的是, 这两个例子都涉及某种 “压缩”, 但原理大不相同

### K-means

1. 二维数据上的 K-means
  - move centroid: 
    - `for k = 1:K`
    - `centroids(k, : ) = mean(X(find(idx == k), : ));`
2. 用 K-means 把图片压缩成 16 色
  - Octave 读入图片时 得到一个 rank-3 的矩阵. 三个维度分别为: 行, 列, RGB. 
    - 每个像素的单色强度分别用一个 8-bit 整数表示. 对 128x128 像素图片, 矩阵形状为 128x128x3.
  - reshape 为 16384x3 的矩阵
    - 即 16384 个样本. 注意: 这个例子里的样本是像素点而非图片!
    - 数据是三维: R, G, B
  - 对这个矩阵实施 K-means, 找到 16 个 centroid. (通过聚类, 找到16种典型颜色)
  - 最后每个像素只需要 4-bit 来存储对应的 centroid 编号. 图片大小减小到约 1/6.

### PCA

1. 二维数据降至一维的例子
2. 对人脸图片使用 PCA
  - 原图片是 32x32 像素, 1024 个特征
  - 用 PCA 降至 100 维, 图片丢失了一些细节, 但整体上清晰可辨.
  - 这种处理可以显著加快学习速度. 训练神经网络前可以采用.
3. PCA 用于可视化: 把三维数据转为二维
  - The PCA projection can be thought of as a rotation that selects the view that maximizes the spread of the data, which often corresponds to the “best” view.


## w9 Anomaly detection

### Density estimation

- 例子:
  - Fraud detection
  - Manufacturing. 检测产品质量
  - monitoring computers in a data center
- 模型: $p(x) = \prod_{j=1}^n p(x_j; \mu_j, \sigma_j^2)$
  - 上式的前提是独立性假设. 但实际中, 即便各特征并不相互独立, 问题一般不会很大.
  - 这类问题叫做 **density estimation**
- 算法
  - 模型训练就是估计各 p(x_i) 的均值和方差 

![](https://d2mxuefqeaa7sj.cloudfront.net/s_C71F128EC1AD0AA0CC664FF1B2FF7C9CC5A870050A4FC3E2D26DBB5320946F3D_1501316430264_image.png)


### Building an anomoly detection system

- developing and evaluating an anomaly detection system
- anomaly detection 是一个无监督学习算法. 但为了评估算法表现, 需要少量标记数据


- 训练/CV/测试集划分
    - 一般会有大量的”正常”数据, 另外需要少量”反常”数据 (比如 20-50个)
    - 假设 10000 个正常数据, 20 个反常数据. 可划分如下:
      - 训练集: 6000 个正常数据 (可以允许混入少量反常数据).
      - 交叉验证集: 2000 正常, 10 反常
      - 测试集: 2000 正常, 10 反常
- 训练: 无监督学习, 不需要 label
  - 交叉验证/测试: 
    - 需要 label 来评估模型, 使用 presision/recall/f1-score 等指标. 
    - 交叉验证集可用来决定特征的选取, 超参数 $\epsilon$ 的取值等
- 如果没有标记数据: 依然可以学习 p(x), 但难以评估模型表现, 难以选择合适的 $\epsilon$
- anomaly detection vs. supervised learning
  - 当 异常样本 数量很少时, 无法用监督学习方法, 只能用 anomaly detection 方法
  - anomaly detection 没有办法学到关于”异常”的具体知识, 未来遇到的异常样本与数据集里已有的异常样本很可能并不相似.
- choosing what features to use
  - 处理 non-gaussian features
    - non-guassian features 直接拿来用也不是不可以. 但如果能处理一下, 效果会更好.
    - 常见的处理: 取 log, 开 n 次根号, 等. 见下图演示.

![](https://d2mxuefqeaa7sj.cloudfront.net/s_C71F128EC1AD0AA0CC664FF1B2FF7C9CC5A870050A4FC3E2D26DBB5320946F3D_1501320374474_image.png)

- 依靠 error analysis 来增加新的特征
    - 当现有算法无法识别某些异常点时, 试试引入什么新的特征可以把这些点区分开来.

### Multivariate Gaussian Distribution

- 多维高斯分布
  - 不能分解为各特征分布的乘积. 因此直接对联合分布建模 $p(x; \mu, \Sigma)$
  - $\mu \in R^n,\ \Sigma \in R^{n\times n}$ ← 协方差矩阵
    - 相互独立时, $\Sigma$ 是 对角阵 $(\sigma_1^2,...,\sigma_n^2)$
  - 幻灯里有二维高斯分布当协方差矩阵为不同情况时 p(x1, x2) 的三维图和 contour plot.
    - 非对角元素绝对值越大, 两个变量相关性越强
- 使用多维高斯分布实现异常检测
  - 参数估计

![](https://d2mxuefqeaa7sj.cloudfront.net/s_C71F128EC1AD0AA0CC664FF1B2FF7C9CC5A870050A4FC3E2D26DBB5320946F3D_1501322305251_image.png)

- 概率连乘模型 or 多维高斯分布模型?
  - 前者应用更广.
  - 当特征存在相关性时:  使用前者需要手动创建特征来考虑相关性 (如 x_new = x2/x1). 后者可以自动捕捉相关性.
  - 参数数量: 前者是 2n, 后者是 ~n^2/2. 前者计算代价小的多. 后者需要对 nxn 矩阵求逆, 当 n 较大时计算量很大.
  - 前者在 m 较小时也 ok. 后者需要 m > n, 否则协方差矩阵不可逆 (建议 m >= 10n).
    - 协方差矩阵不可逆的两个原因: 1. m < n, 2. 有冗余特征 (即线性相关的特征)
  - 简言之, **如果 n 不太大, m >= 10n, 并且希望模型自动考虑特征相关性, 那就用多维高斯分布模型.** 否则别用.


## w9 Recommender system

### 引子: content-based recommendations

- 按电影内容取不同特征, 每部电影的特征向量 $x^{(i)}$
- 每个用户取不同参数向量 $\theta^{(j)}$
- 用 r(i, j) 记录用户j是否对电影i评了分.
- content-based 方法就是: 给定 $x^{(i)}$, 学习 $\theta^{(j)}$. 
- 相当于许多个线性回归问题. 目标函数是这些线性回归 square error 之和. 这里 square error 不取平均 (可能是为了方便求和)

![](https://d2mxuefqeaa7sj.cloudfront.net/s_C71F128EC1AD0AA0CC664FF1B2FF7C9CC5A870050A4FC3E2D26DBB5320946F3D_1501816082520_image.png)

- 问题是实际中不容易事先确定电影的特征向量

### Collaborative filtering

- 换一个角度: 给定 $\theta^{(j)}$, 学习 $x^{(i)}$ 



![](https://d2mxuefqeaa7sj.cloudfront.net/s_C71F128EC1AD0AA0CC664FF1B2FF7C9CC5A870050A4FC3E2D26DBB5320946F3D_1501816023957_image.png)

- 可以用迭代的方法: 随机初始化 theta, 然后依次优化 x → theta → x → …
- 效率更高的方法: 随机初始化, 然后同时更新 theta 和 x (前面两种目标函数里的误差平方和项是相同的)
  - ? 这里不再保留 x_0 = 1 的 convention
  - 随机初始化可打破对称性
  - 这一算法可以自主学习特征. 但最终学到的特征不容易解读.

![](https://d2mxuefqeaa7sj.cloudfront.net/s_C71F128EC1AD0AA0CC664FF1B2FF7C9CC5A870050A4FC3E2D26DBB5320946F3D_1501816737931_Screenshot+2017-08-04+11.17.32.png)

- 梯度计算

![](https://d2mxuefqeaa7sj.cloudfront.net/s_C71F128EC1AD0AA0CC664FF1B2FF7C9CC5A870050A4FC3E2D26DBB5320946F3D_1501839327267_image.png)

- 其实跟线性回归很像. 区别: square error 没有求平均; 求和的范围是 r(i,j) = 1 的样本.
- 作业里的例子
  - 求梯度的向量化语句写起来比较 tricky. 以 x 的梯度为例:
     for i = 1:num_movies
       idx = find(R(i, :) == 1);
       Theta_temp = Theta(idx, :);
       Y_temp = Y(i, idx);
       X_grad(i,:) = (X(i, :) * Theta_temp' - Y_temp) * Theta_temp + lambda * X(i, :);
     end
  - 向量化的技巧需要再学习. 
    - 首先, 需要能识别出某些常见的 pattern, 可以直接写成比较简洁的向量化表达, 比如 `A * B` 或 `A * x`. 
    - 然后, 有些情况下, 可能没法直接写成最简洁的向量化形式, 需要搭配循环. 同时, 借助于 slicing / 查找index 等手段来分解问题.

### Low rank matrix factorization

- 评分矩阵 Y: $n_m \times n_u$. 行对应电影, 列对应用户
- 矩阵分解: $Y = X\Theta^T$. 其中 X, $\Theta$ 分别为以 x 和 $\theta$ 为行向量的矩阵.
  - 不明白这种方法如何应用? 因为矩阵 Y 里面有很多未知项
- 得到不同电影的特征向量 x 后, 可以用 $||x^{(i)} - x^{(j)}||$ 衡量相似性, 据此可以做推荐.

### mean normalization

- 背景: 有的用户没有评过任何电影. collaborative filtering 的目标函数里, 这类用户的参数向量只存在于正则化项中, 所以最后学到的结果是0向量, 意味着他对任何电影都打0分. 这不太合理.
- 解决: 将评分矩阵 Y 对每部电影分别做 mean normalization. 提取出各电影的平均分向量 $\mu$. 最终用户j对电影i评分的预测值为 $(\theta^{(j)})^T x^{(i)} + \mu_i$. 这样, 未评分用户的预测评分等于电影平均分.
- 扩展: 另一个角度是, 对每个用户分别做 mean normalization, 从而矫正无人评分的电影的预测评分. 不过这种情况在实际应用中不重要, 因为无人问津的电影不值得推荐.


## w10 Large Scale Machine Learning
- 这一周介绍与大量数据打交道的两个办法: 一是 SGD 和 mini-batch. 二是 map reduce.
- 不要一上来就用大数据集. 先从小数据集开始, 做出 learning curve.

### SGD 和 mini-batch

- SGD 中遍历训练集的次数与训练集大小有关, 典型的是 1-10次
- mini-batch 的 batch size 典型值是 2-100.
- 如何监控 SGD 的收敛性
  - 每隔 k 个 iteration, 计算出过去 k 个样本的平均 cost (注意: 不需要计算在整个数据集上的 cost).
  - k 偏小时, 曲线震荡会比较剧烈, 可能看不出趋势. k 偏大时, 得到的反馈没那么及时. 典型取值可能是 数千.
- 学习率: 可以固定, 也可以缓慢减小. 如 $\alpha = \frac{\mathrm{const1}} {\mathrm{iterationNumber} + \mathrm{const2}}$

### online learning

- 把用户的实时反馈作为 SGD 的训练样本. 可以适应用户偏好随时间的变化.
- 可以用于: 定价策略优化, 点击率, 商品推荐, 等.

### map reduce and data parellelism

- map reduce: 让多台机器同时参与一个训练过程. 
  - master 获得各机器算出的梯度, 求和并更新参数.
  - 适合 map reduce 的算法: 核心计算过程是某种在数据集上的求和, 比如当梯度下降中 cost 和梯度里是 sum 的形式.
- 类似地, 同一台机器上也可以把计算分摊到不同的 Core
- 如果算法本身是向量化的形式, 而利用的线性代数模块又支持多核计算, 那么可能就不用自己去操心这个问题


## w11 application example: photo OCR
- 用 pipeline 来分解和组织机器学习任务
- Photo OCR pipeline: text detection → character segmentation → character classification

### sliding window

- text detection
  - 先讲一个行人检测的例子
    - 先定一个长宽比, 如 82x36, 先找一堆这样大小的正负例图片去训练 window 分类器
    - 用 82x36 的 window 去扫描目标图片, 取一个合适的 step size.
    - 再逐渐增大 window 尺寸继续扫描. 大 window 里的图像会被压缩至 82x36 来分析.
  - 文字检测更困难一些, 因为 window 长宽比不确定
    - 可能需要用不同长宽比的 window 去扫描
    - 分类器训练: 正例 - 有文字, 负例 - 无文字
    - 之后可以有一步是 expansion, 把包含文字的相近小区域连成片. 然后根据长宽比去除瘦高的区域.
- character segmentation
  - 先训练 window 分类器. 与前一步不同, 这里分类器的目标是找到字符分离的边界. 下图左侧为正例.

![](https://d2mxuefqeaa7sj.cloudfront.net/s_C71F128EC1AD0AA0CC664FF1B2FF7C9CC5A870050A4FC3E2D26DBB5320946F3D_1501901611182_image.png)


### artificial data synthesis

- 人工合成数据, 两种情况: 1. 直接生成, 2. 由已有的真实数据变化而来
- 为文字识别分类器准备人工数据:
  - 可以从不同字体库选文字, 加上背景及其他处理
  - 可以对真实文字图片进行变形处理 (distortions)
- 对语音数据进行变形处理: 用信号差的手机播放, 加噪音, 等等.
- 这种 distortion 处理需要注意: 引入的 noise 或 distortion 应该是在测试集上有代表性的. 而若是加入随机的或无意义的噪音, 一般没用.
- 对使用更多数据的建议
  - 先通过学习曲线等办法, 确保目前分类器是 low bias. 如果不是, 可引入新特征, 增加模型复杂度.
  - 对自己和团队提问: 要想获得比手头多10倍的数据, 需要多少工作量? 答案可能比你的预期更乐观. 可考虑的方法:
    - 人工合成数据
    - 自己动手收集和标记数据.
    - crowd source. 花钱找人标数据. 缺点是数据质量不一定高.

### ceiling analysis

- 更有效地分配精力和资源 (功能类似的工具: learning curve, error analysis)
- 假设 pipeline 上有三个部分 ABC. 依次把 A, A&B, A&B&C 的输出替换为”正确值”, 看最终指标提高的幅度有多大.













