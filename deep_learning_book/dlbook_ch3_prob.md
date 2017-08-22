# Ch3 Probability and InformationTheory
http://www.deeplearningbook.org/contents/prob.html

## 目录

<!-- toc -->

- [3.1 Why Probability](#31-why-probability)
- [3.2~3.8 一些概率论基础知识](#3238-%E4%B8%80%E4%BA%9B%E6%A6%82%E7%8E%87%E8%AE%BA%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86)
- [3.9 Common probability distributions](#39-common-probability-distributions)
- [3.10 Useful properties of common functions](#310-useful-properties-of-common-functions)
- [3.13 Infomation theory](#313-infomation-theory)
- [3.14 Structured probabilistic models](#314-structured-probabilistic-models)

<!-- tocstop -->

## 3.1 Why Probability
- 计算机科学的大多数分支都不像机器学习这样依赖概率论, 因为机器学习的研究对象具有 uncertainty 和 stochasticity.
- 世间万物往往都有不确定性. 不确定性的三个来源:
  - 系统本身的随机属性
    - 如, 量子力学尺度上的粒子; 某些我们假定的随机场景, 比如掷骰子或发牌
  - 观测的局限性 incomplete observability
    - 即便是 deterministic systems, 因为我们无法观测到影响系统行为的所有变量, 因此也会表现出随机性.
    - 如 [Monty Hall problem](https://en.wikipedia.org/wiki/Monty_Hall_problem), 每扇门背后的东西是事先确定的, 但参与者看不到
  - 模型的局限性 incomplete modeling
    - 某些信息可以观测到, 但建模时被丢弃, 也会造成模型预测的不确定性
    - 比如有些观测量本来是连续的, 但最后模型输出离散值
- 很多时候 **简单但不确定**的规则 比 复杂而确定的规则 更实用
  - 比如 “Most birds fly”, 若想改为确定的形式, 会非常复杂, 得不偿失: “Birds ﬂy, except for very young birds that have not yet learned to ﬂy, sick or injured birds that have lost the ability to ﬂy, ﬂightless species of birds including the cassowary, ostrich and kiwi. . .” :D
- frequentist probability vs. Bayesian probability
  - frequentist probability: 用于可重复的事件. 概率等于重复无穷多次时事件发生的比例. 概率论最初就是用来分析此类问题的.
  - Bayesian probability: 不可重复的事件. 概率代表确定的程度 (degree of belief).
  - 尽可能使概率论的框架同时适用于这两类事件.
- 概率可看作逻辑在不确定性问题上的延伸


## 3.2~3.8 一些概率论基础知识
- 主要注意下数学记法
- 数学符号的字体差别:
  - 随机变量 $\mathrm{x}$ ~  `\mathrm{x}`
  - 随机变量取值 $x_1, x_2, ...$
  - 随机向量 $\mathbf{x}$ ~  `\mathbf{x}`
  - 随机向量取值 ~ 粗斜体 x
- Discrete variables and probability mass functions
  - $P(\mathrm{x} = x)$ 常简写为 $P(x)$
  - 随机变量 $\mathrm{x}$ 服从某种 PMF: $\mathrm{x} \sim P(\mathrm{x})$
- Continuous variables and probability density functions
  - 有参数时 $p(x;a,b)$ ~ 分号表示 “parametrized by”
- independence and conditional independence
  - 分别记做 $\mathrm{x} \perp \mathrm{y}$ , $\mathrm{x} \perp \mathrm{y} | \mathrm{z}$
  - 注意这二者相互都不能推出
## 3.9 Common probability distributions
- Bernoulli distribution: 单参数 $\phi$ , $Var(x) = \phi (1-\phi)$
- Multinouli disdtribution: k个分类 有 k-1 个参数. 
  - multinoulli 分布是 multinomial 分布的一种特殊情形 (n=1). 但经常有人混用这两个术语.
- Gaussian distribution
  - In the absence of prior knowledge about what form a distribution over the real numbers should take, the normal distribution is a good default choice for two major reasons.
    - 现实中很多分布都接近正态分布. 中心极限定理表明, 多个独立随机变量之和近似服从正态分布.
    - ?? 当方差相同时, 在所有可能的分布中, 正态分布 encodes the maximum amount of uncertainty over the real numbers, 因而 inserts the least amount of prior knowledge into a model.
- exponential and Laplace distributions

![](https://d2mxuefqeaa7sj.cloudfront.net/s_9A8B278A347D550D9B4FD3B13A92C5CF9D4BF67E5696FCFB930654FE5B66B082_1500434865187_image.png)

- the Dirac distribution and empirical distribution
  - the Dirac delta function $\delta(x)$:  一种特别的函数
    - $\delta(x) = 0$ 当 $x \neq 0$
    - $\int_{-\infty}^{\infty} \delta(x) = 1$
  - Dirac distribution: $p(x) = \delta(x-\mu)$
    - 可看做是正态分布当方差趋于无穷小时的极限
  - empirical distribution
    - 对连续变量 x, 可以借助 Dirac distribution 定义: $\hat{p}(x) = \frac{1}{m} \sum_{i=1}^m \delta(x - x^{(i)})$
    - 对离散变量 x, empirical distribution 就像是 multinoulli distribution
    - 两种解读
      - ?? 训练集数据可构成一个 empirical distribution
      - empirical distribution 可最大化训练数据的 likelihood
- Mixture distribution
  - $P(x) = \sum_i P(c=i) P(x|c=i)$
    - $P(x|c=i)$ 代表不同的 component distributions
    - P(c) 代表从不同 component 中选择, 是一个 multinoulli distribution
  - c 是一种 latent variable, 一种无法直接观测的随机变量.
  - 常见的一种 mixture model: **the Gaussian mixture model** → **a universal approximator of densities**


## 3.10 Useful properties of common functions
- sigmoid
  - commonly used to produce the $\phi$ parameter of a Bernoulli distribution
- softplus $\zeta(x) = \log (1 + \exp(x))$
  - 是 max(0,x) 函数的一种 soft version
  - 跟 sigmoid 有密切关系
- 有用的性质
  - $\sigma(x) = \frac{\exp(x)}{\exp(x) + \exp(0)}$
  - $\frac{d}{dx} \sigma(x) = \sigma(x) (1 - \sigma(x))$
  - $1 - \sigma(x) = \sigma(-x)$
  - $\log \sigma(x) = - \zeta(-x)$
  - $\frac{d}{dx} \zeta(x) = \sigma(x)$
  - $\forall x\in (0,1), \sigma^{-1}(x) = \log(\frac{x}{1-x})$  ← logit
  - $\forall x > 0, \zeta^{-1}(x) = \log(\exp(x) - 1)$
  - $\zeta(x) = \int_{-\infty}^x \sigma(y)dy$
  - $\zeta(x) - \zeta(-x) = x$
## 3.13 Infomation theory
- 信息论的简介和 basic intuition
  - infomation theory: revolves around quantifying how much information is present in a signal.
  - In this textbook, we mostly use a few key ideas from information theory to **characterize probability distributions** or **quantify similarity between probability distributions.**
  - **The basic intuition behind information theory**: learning that an unlikely event has occurred is more informative than learning that a likely event has occurred. 概率越小的事件信息量越大.
  - quantify information in a way that formalizes this intuition
    - 可能性较大的事件 信息量小. 必然发生的事件 没有信息量.
    - 可能性较小的事件 信息量大.
    - 彼此独立的事件, 总信息量应该是相加的. ← 这是取对数的目的.
- **self-information** of an event $\mathrm{x} = x$: $I(x) = - \log P(x)$
  - 本书中对数以 e 为底. 由此得到的单位是 nats. 以 2 为底得到的才是 bits.
  - self-information 仅仅针对 single outcome
- **Shannon entropy**: 衡量整个概率分布的不确定性
  - $H(x) = \mathbb{E}_{x\sim P}[I(x)] = -\mathbb{E}_{x\sim P}[\log P(x)]$ . 也可记做 $H(P)$
  - It gives a lower bound on the number of bits/nats needed on average to encode symbols drawn from a distribution P.
  - 分布接近 deterministic 时, entropy 较小. 分布接近 uniform 时, entropy 较大. 比如参数为 p 的二项分布, $H(P) = (p-1)\log(1-p) - p\log p$, 如下图

![](https://d2mxuefqeaa7sj.cloudfront.net/s_9A8B278A347D550D9B4FD3B13A92C5CF9D4BF67E5696FCFB930654FE5B66B082_1500449643902_image.png)

- **KL divergence**: 衡量两个概率分布的差别
  - $D_{KL}(P||Q) = \mathbb{E}_{x\sim P} [\log \frac{P(x)}{Q(x)}] = \mathbb{E}_{x\sim P}[\log P(x) - \log Q(x)]$
  - 物理意义: In the case of discrete variables, it is the extra amount of information needed to send a message containing symbols drawn from probability distribution P, when we use a code that was designed to minimize the length of messages drawn from probability distribution Q.
  - 类似某种距离. 但 KL divergence 不是对称的.
- **cross-entropy**
  - $H(P,Q) = H(P) + D_{KL}(P||Q) = -\mathbb{E}_{x\sim P} \log Q(x)$
    - 注意 P 在期望的下标里, 在求期望过程中起作用
  - 与 KL divergence 的区别是少了一项 H(P). 关于 Q 最小化交叉熵 等价于 最小化 KL divergence.
- notes
  - 线索: 单个事件的信息量 (求期望) → 一个概率分布的信息量/不确定性 → 两个概率分布的差别
  - 离散的情况下的(香农)熵和交叉熵
    - $H(P) = -\sum_{x\sim P} P(x_i) \log P(x_i)$
    - $H(P,Q) = -\sum_{x\sim P} P(x_i) \log Q(x_i)$
    - 其中 $P(x_i)$ 项是求期望产生的. log 项来自单个事件的 self-information 定义.
  - 关于香农熵 H(P)
    - 当概率分布为均匀分布时 (P(x_i) = 1/n) , H(P) = log n. 随着选项的数量 n 增大, 香农熵以对数速率增大.
    - 当选项数量一定时, 分布越均匀, 香农熵越大.
      - 上限: 均匀分布时 H(P) = log n. 
      - 下限: 某一选项概率趋于1时, H(P) → 0. 


## 3.14 Structured probabilistic models
- 背景: 机器学习中的概率分布涉及的随机变量很多, 而随机变量的相互作用往往只存在于小范围. 如果使用单一的联合概率分布函数, 会比较低效 (both computationally and statistically)
- 对策: 我们可以把概率分布分解为多个因子的乘积. 把不相关的随机变量分离开.
- 用 graphs 表示概率分布的分解 → **structured probabilistic model / graphical model**
- directed models
  - 分解出来的因子都是条件概率分布
- undirected models
  - 两两相关的节点集合称为 clique
  - 分解出来的因子是不同 clique 的非负函数, 但不是概率分布
- these graphical representations of factorizations are a language for describing probability distributions. 只是一种描述方式.

