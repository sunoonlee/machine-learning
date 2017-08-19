# Ch5 machine learning basics
- http://www.deeplearningbook.org/contents/ml.html

[x] pre-read 5.1-5.4
[x] 5.1 Learning Algorithms
[x] 5.2 Capacity, Overfitting and Underfitting 
[x] 5.3 Hyperparameters and Validation Sets
[x] 5.4 Estimators, Bias and Variance
[x] 5.5 Maximum Likelihood Estimation 
[x] 5.6 Bayesian Statistics 
[ ] 5.7 Supervised Learning Algorithms 
[ ] 5.8 Unsupervised Learning Algorithms 
[ ] 5.9 Stochastic Gradient Descent
[ ] 5.10 Building a Machine Learning Algorithm 
[ ] 5.11 Challenges Motivating Deep Learning


## Deep learning book organization

![](https://d2mxuefqeaa7sj.cloudfront.net/s_E7D57D4995ADAADB631B2CD3538497B166B99657FF18AE0CC06279D9388CFD38_1497927934869_image.png)

## Ch5 intro

机器学习: 可看做一种”应用统计学”

> Machine learning is essentially a form of applied statistics with increased emphasis on the use of computers to **statistically estimate complicated functions** and a decreased emphasis on proving conﬁdence intervals around these functions.

介绍了统计学的两类核心方法

- 频率论 frequentist estimators → 5.4~5.5
- 贝叶斯派 Bayesian inference → 5.6


## 5.1 Learning Algorithms

什么是学习? Mitchell 的经典定义

- 三要素: 任务 T, 指标 P, 经验(数据) E
- 学习 = 靠 E 提高 T 的 P

### Task

- 介绍了一些常见的机器学习任务. 几个例子:
  - classification with missing inputs: 如医疗诊断领域, 有些检查代价比较大.
  - anomaly detection. 比如信用卡欺诈. 盗刷的消费行为与正常行为的概率分布不同.
  - synthesis and sampling: 合成与采样. 用于多媒体领域, 比如生成游戏中的大块纹理. 可以节省艺术工作者的劳动.
  - density estimation or probability mass function estimation. 

### Performance measure

- 好的指标有时并不好找
  - 有时, 难以决定该测量什么. 比如翻译/对话问题.
  - 有时, 我们虽知道一个“理想”的指标, 但难以测量. 比如 density estimation 问题中, 难以准确计算特定点的概率值. 这时只能退而求其次.

### Experience

- 根据数据的不同, 可以对机器学习分类
  - 根据数据中有无 label/target, 可以分为监督/无监督学习.
    - 无监督学习经常是要学习 p(x) 的分布, 或者该分布的一些特性.
    - 而监督学习通常是要学习 p(y|x)
    - 二者没有严格界限. 比如 通过把 p(x) 分解成 x各分量条件概率的连乘, 相当于把无监督学习问题转化为了数个监督学习问题.
  - 其他分类: 半监督学习, multi-instance learning, 强化学习
- dataset 的表示: a design matrix. 本书中, 样本按行向量处理


## 5.2 Capacity, Overﬁtting and Underﬁtting

### 机器学习的核心挑战: 泛化

- 粗略地说, 机器学习 = 优化问题 + 泛化问题
- 泛化的可能性?
  - statistical learning theory 这个领域给出了一些答案
  - 必要的假定: i.d.d. assumptions - 样本彼此独立; 训练集与测试集同分布.
    - 背后的这个分布叫 “data generating process/distribution”

### 模型的 capacity

- 即模型拟合函数的能力
- how to control the capacity of a learning algorithm
  - 选择不同的假设空间, 即改变模型的 “**representational capacity**”
    - 由于一些制约因素(比如优化算法的局限), 实际学习算法的 **effective capacity** 可能低于模型的 representational capacity.
  - 假设空间的大小和假设函数的性质都会影响 capacity
  - 另一种方法是 regularization
- capacity 需要同时匹配 任务复杂度 和 数据量, 机器学习才能得到比较好的结果
- VC dimension: 衡量模型 capacity 的指标之一
  - 可由此得到一个 generalization bound
  - 但深度学习实践中很少用这个 generalization bound
    - 一方面, bound 太松了用处不大
    - 另一方面, 深度学习模型的 capacity 不易确定.
      - 有效 capacity 受限于优化算法的能力
      - non-convex 优化的理论还不成熟
      > The problem of determining the capacity of a deep learning model is especially diﬃcult because the eﬀective capacity is limited by the capabilities of the optimization algorithm, and we have little theoretical understanding of the very general non-convex optimization problems involved in deep learning.
- capacity 和 error 的典型关系: 
  - generalization error 常常与 capacity 成 U 形关系, 最低点就是 overfit/underfit 临界点

![](https://d2mxuefqeaa7sj.cloudfront.net/s_E7D57D4995ADAADB631B2CD3538497B166B99657FF18AE0CC06279D9388CFD38_1497943050516_image.png)

- high capacity 的极端: non-parametric models
  - 一种 “practical non-parametric model”: nearest neighbor regression

### Bayes error

- 一个知道”真实”概率分布的模型预测时仍会有误差 → Bayes error
- 等同于 LFD 课程里讲的 stochastic noise
- 是 generalization error 的下限

### The No Free Lunch Theorem

- bad news: *averaged over all possible data generating distributions*, every classiﬁcation algorithm has the same error rate when classifying previously unobserved points
- good news: If we make assumptions about the kinds of probability distributions we encounter in real-world applications, then we can design learning algorithms that perform well on these distributions.
  - → we must design our machine learning algorithms to perform well **on a speciﬁc task**
- 结论是机器学习应该是针对具体问题和任务, 而不是追求通用答案

### Regularization

- expressing preferences for different solutions
- “Regularization is any modiﬁcation we make to a learning algorithm that is intended to reduce its generalization error but not its training error.”


## 5.3 Hyperparameters and Validation Sets

### 超参数

- 超参数: 不是由算法自己学来的那部分”参数”
- 适合作为超参数而不是参数的两种情况:
  - 若作为参数, 对学习算法来说很难 optimize
  - **这个”参数”不宜从训练集学得** ← 任何控制模型 capacity 的超参数都是如此
    - 从训练集学习的话, 它只管 train error, 不管 generalization error. 结果总是给你选 capacity 最大的, 很容易 overfit.

### validation set

- test set 不能碰. 一般从 training data 里分出 validation set. 典型的比例是 8:2.
- validation set error 常常比 generalization error 略低.
- 用于测试的 benchmark datasets 因为大家都在用, 也会被污染, 所以也需要及时更新.

### cross-validation

- 当总的数据量有限时, 分出来的 test set 可能很比较小, 这时 test 结果的不确定性会比较大.
- 这时可以利用 cross-validation, 以增加计算量为代价来提高 test 结果的可信度.
- 常见的方法是 k-fold cross-validation procedure. 流程见书中 P123.


## 5.4 Estimators, Bias and Variance
- 用统计学中的概念 — 参数估计, bias 和 variance — 来帮助理解泛化/欠拟合/过拟合问题.

### Point Estimation

- point estimator 定义: data 的任意函数 $\hat{\theta}_m = g(x^{(1)},...x^{(m)})$. 目的是准确预测参数的”真实值”.
  - 用于估计的 data 是依据某种分布随机生成的, 因此 estimator 是一个**随机变量**
- 除了估计参数, 有时直接估计函数: approximate $f$ with $\hat{f}$
- 最常见的 estimator 就是样本均值
- 下面介绍 estimator 的两个重要特性: bias, variance

### Bias

- $\mathrm{bias}(\hat{\theta}_m) = \mathbb{E}(\hat{\theta}_m) - \theta$
- unbiased: $\mathrm{bias}(\hat{\theta}_m) = 0$
- asymptotically unbiased: $\lim_{m\rightarrow \infty} \mathrm{bias}(\hat{\theta}_m) = 0$
- 例子
  - 样本均值是对总体均值的无偏估计. 
    - test error 是对 generalization error 的无偏估计. 而 validation error 不是.
  - 高斯分布的 variance parameter $\sigma^2$ 的两种 estimator:
    - 偏差平方和/m: bias = $-\sigma^2/m$
    - 偏差平方和/(m-1): 无偏估计

### Variance and Standard Error

- $\mathrm{Var}(\hat{\theta})$ 和 $\mathrm{SE}(\hat{\theta})$ . A measure of how we would expect the estimate we compute from data to vary as we independently resample the dataset from the underlying data generating process.
- 主要应用: **standard error of the mean**
  - 即 “sample mean” 这个 estimator 的 standard error
  - $SE(\hat{\mu}_m) = \sigma / \sqrt{m}$  . 于是可转而用 $\sigma$ 的估计量来表示.
  - 如何估计 $\sigma$
    - 记 $S_{N-1}^2 = \frac{1}{N-1} \sum_i (x_i - \hat{\mu}_m)^2$. 则 $\sigma^2 = E(S_{N-1}^2) = Var(S_{N-1}) + (ES_{N-1})^2 \geq (ES_{N-1})^2$
    - 故 $E(S_{N-1}) \leq \sigma$ . $S_{N-1}$ 不是 $\sigma$ 的无偏估计. (尽管 $S_{N-1}^2$ 是 $\sigma^2$ 的无偏估计)
    - $S_N$ 或 $S_{N-1}$ 都不是 $\sigma$ 的无偏估计. 其中 $S_{N-1}$ 的 bias 相对较小.
    - 当样本量 m 很大时, the approximation is quite reasonable
  - 在机器学习上的应用:
    - 常用 test error 的 sample mean 来估计 generalization error. 根据中心极限定理, sample mean 近似服从正态分布.
    - 可以利用 standard error of the mean 来计算真实值落在某一区间的概率
    - 95% 置信区间: $( \hat{\mu}_m - 1.96SE(\hat{\mu}_m), \hat{\mu}_m + 1.96SE(\hat{\mu}_m) )$
    - 比较机器学习算法优劣
      - 当算法A的 95% 置信区间上界 小于 算法B的 95%置信区间下界时, 可以说 A 比 B 好.

### Trading off bias and variance to minimize MSE

- $\mathrm{MSE} = \mathbb{E}[( \hat{\theta}_m - \theta)^2] = \mathrm{Bias} (\hat{\theta}_m)^2 + \mathrm{Var} (\hat{\theta}_m)$
  - 刚好可以把 error 分解成有意义的两项
  - 衡量 estimator 性能的一个常用指标. 优点是平方函数在数学上较易处理.

### Consistency

- $P(| \hat{\theta}_m - \theta| > \epsilon) \rightarrow 0 \text{ as } m \rightarrow \infty$ . 是一种 “convergence in probability”.
  - 当样本大小无限增加时, 估计量”依概率收敛”于被估计的值.
- 中文叫做 “相合性”. 是一个良好的估计量应具有的基本性质.
- “大样本性质” 与 “小样本性质” (陈希儒-概率论与数理统计, P174)
  - 相合性是在样本数趋于无穷大的背景下定义的. 这类性质称为 “大样本性质”. 类似的还有 “渐进正态性”.
  - 相对地, “无偏性”等概念是对固定的样本大小来说的. 所以是 “小样本性质”.
- consistency 是 asymptotic unbiasedness 的充分不必要条件.


## 5.5 Maximum Likelihood Estimation
- 我们希望利用一些原则来推导出 estimator (而不是靠猜). 最常见的一种原则就是 MLE.
- 首先区分三种分布
  - “真实”概率分布 $p_{data}(x)$
  - 训练集上的 **empirical distribution** $\hat{p}_{data}$
  - 模型的概率分布族 $p_{model}(x;\theta)$
- 参数向量 $\theta$ 的极大似然估计
  - $\theta_{ML} = \mathrm{argmax}_{\theta} p_{model}(X; \theta) = \mathrm{argmax}_{\theta} \prod_{i=1}^m p_{model}(x^{(i)}; \theta)$
    - 训练集 X 已知, 最大化 $p_{model}(X;\theta)$
  - 为方便处理, 取 log: $\theta_{ML} = \mathrm{argmax}_{\theta} \sum_{i=1}^m \log p_{model}(x^{(i)}; \theta)$
  - 除以 m: $\theta_{ML} = \mathrm{argmax}_{\theta} \mathbb{E}_{x\sim \hat{p}_{data}} \log p_{model}(x; \theta)$  → 加上负号就是交叉熵了
- 由上式, 最大化似然函数 相当于最小化 $\hat{p}_{data}$ 与 $p_{model}(x;\theta)$ 的交叉熵, 也相当于最小化二者的差异 (dissimilarity).
  - 这种差异可用 KL divergence 衡量:
    - $D_{KL}(\hat{p}_{data} || p_{model}) = \mathbb{E}_{x\sim \hat{p}_{data}} [\log \hat{p}_{data}(x) - \log p_{model}(x; \theta)]$
  - 很多人把”交叉熵”专门用来指 Bernoulli 或 softmax 分布的 negative log-likelihood. 其实是术语使用不当.
  - **任何包含 negative log-likelihood 的 loss 都是训练集定义的 empirical distribution 与模型定义的 p_model 的交叉熵**
    - 比如 MSE 是 empirical distribution 和 Gaussian 模型的交叉熵
  - maximum likelihood 可看做是 an attemp to make p_model match $\hat{p}_{data}$. (理想情况自然是去 match $p_{data}$, 但对后者我们无从下手)

### Conditional log-likelihood and MSE

- #quiz 通过推导证明: 最小化 MSE 等价于最大化似然函数 (也等价于最小化交叉熵)
  - 提示: 假设 $p(y|x) = N(y;\hat{y}(x;w), \sigma^2)$

### Properties of maximum likelihood

- 极大似然估计的优良特性
  - consistency 相合性 (依概率收敛) ← 需满足一定的前提条件
  - statistic efficiency: 在所有具备相合性的估计量中, 极大似然估计的效率最高, 随样本量增大的收敛速度最快, 达到相同水平泛化误差所需的样本数最少.
- 因此, 极大似然估计在机器学习中比较受青睐.
- 当样本数较小而可能产生过拟合时, 通过正则化得到 a biased version of maximum likelihood, 可以降低 variance.
## 5.6 Bayesian Statistics
- 这一节简单介绍了另一个流派. 贝叶斯派与频率论派的思路有很大不同, 最有争议的区别在于先验概率. 但两派方法得到的结果有一定的联系. 在某些情形下, 先验概率的贡献与正则化类似.
- $p(\theta|data) = \frac{p(data|\theta) p(\theta)}{p(data)}$
- 两大特点: 一是对参数的处理方法不同, 二是先验概率
- 对参数 $\theta$ 的处理方法
  - 频率论
    - 认为参数 $\theta$ 存在固定但未知的“真实”值. 先用点估计去获得一个确定的 $\theta$ 值. 然后用这个 $\theta$ 值去预测.
    - 认为 dataset 具有随机性, 点估计 $\hat{\theta}$ (作为 dataset 的函数) 因此成为一个随机变量.
  - 贝叶斯派
    - 概率反映 knowledge 的不确定性程度. 认为 dataset 是观测到的事实, 不是随机的. 认为 $\theta$ 因其本身的不确定性而成为随机变量.
    - 在预测时考虑 $\theta$ 的所有可能取值, 即利用 $\theta$ 的整个概率分布, 需要对 $\theta$ 取积分. 不同取值的权重就是 $\theta$ 的后验概率分布.
    - 由已有的 m 个样本预测新样本:

![](https://d2mxuefqeaa7sj.cloudfront.net/s_E7D57D4995ADAADB631B2CD3538497B166B99657FF18AE0CC06279D9388CFD38_1500526747932_image.png)

- 先验概率
  - 用先验概率 $p(\theta)$ 表示在观测到数据之前对 $\theta$ 的 knowledge. 
  - 先验概率分布常选为一个熵较大的分布, 如均匀分布或高斯分布. 另一方面, 很多先验概率的选择反映了对更简单或平滑的 solution 的偏好.
  - 批评者认为先验概率引入了人为的主观判断.
- 性能 (相比频率论方法)
  - 训练数据有限时, 泛化性能常常更好.
  - 训练数据较多时, 计算代价更大.

### 贝叶斯方法的线性回归

- 训练数据记为 $X, y$
- y 的条件分布: 假定为高斯分布
  - $p(y|X,w) = N(y;Xw,I) \propto \exp(-\frac{1}{2} (y-Xw)^T(y-Xw))$ 
- 先验分布: 假定为高斯分布 
  - $p(w) = N(w;\mu_0,\Lambda_0) \propto \exp(-\frac{1}{2}(w-\mu_0)^T \Lambda_0^{-1} (w-\mu_0))$
- 参数 w 的后验分布:
  - $p(w|X,y) \propto p(y|X,w) p(w) \propto \exp(\text{一长串})$
  - 引入 $\Lambda_m, \mu_m$, 可得 $p(w|X,y) \propto \exp(-\frac{1}{2}(w-\mu_m)^T \Lambda_m^{-1} (w-\mu_m))$
- 当 $\mu_0, \Lambda_0$ 取某种特殊值时, 结果相当于 MSE + weight decay.

### MAP Estimation

- 贝叶斯方法因为需要 $\theta$ 的整个分布, 处理起来可能比较困难.  ← full Bayesian inference
- 一种简化是: 根据后验概率最大化得到一个 $\theta$ 的点估计.  ← MAP Bayesian inference
- $\theta_{MAP} = argmax_{\theta} p(\theta|x) = argmax_{\theta} [\log p(x|\theta) + \log p(\theta)]$
  - 跟极大似然相比, 最大化的目标多了一项先验概率的对数 $\log p(\theta)$, 这一项会增加 bias, 降低 variance.
  - 当先验分布为 $N(w;0,\frac{1}{\lambda}I^2)$ 时, log-prior 项刚好就是 $\lambda w^Tw$
  - 可以根据这个模式来设计一些复杂的正则化项