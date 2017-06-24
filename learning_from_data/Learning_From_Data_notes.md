# ml_LearningFromData_lim

- Course materials: [Lectures](https://work.caltech.edu/lectures.html)  |  [Homeworks](https://work.caltech.edu/homeworks.html)
- 部分 Homework 解答和笔记: [homeworks.ipynb](https://github.com/sunoonlee/machine-learning/blob/master/learning_from_data/Homeworks.ipynb)
- 推荐安装 chrome 插件 [GitHub with MathJax](https://chrome.google.com/webstore/detail/github-with-mathjax/ioemnmodlmafdkllaclgeombjnmnbima), 支持 Github 上的 LaTeX 公式渲染.

## Lec 1 - The Learning problem


![img](https://d2mxuefqeaa7sj.cloudfront.net/s_FA9189E64E3F201143EF4ED387BEEA610619552CD2E7E2AA295C880EFD385019_1496033144892_image.png)


逻辑线索: what is learning → can we learn → how to do it → how to do it well


- the essence of machine learning
  - a pattern exists
  - we cannot pin it down mathematically
  - we have data on it

**components of learning**

- target function $f$ vs. final hypothesis $g$
  - $f$ is “ideal” and **unknown**. $g$ 是对 $f$ 的近似. $g\in H$ (hypothesis set)
- hypothesis set + learning algorithm = `the learning model`

**a simple model: perceptron**

- 引入 $x_0$ = 1, 则 $h(x) = sign(w^Tx)$
- perceptron 学习算法的 intuition: 向量加法, 向量夹角

**types of learning**

- basic premise of learning
  - **“using a set of observations to uncover an underlying process”**
    - in statistics:
      - underlying process = probability distribution
      - observations = samples
- unsupervised learning: given only input. a way of getting a higher-level representation of the input. 如, 听广播学外语.

**misc**

- learning vs memorizing (overfitting)
- bottleneck of machine learning: generalization


## Lec2 - Is Learning Feasible?

**probability to the rescue 概率拯救”学习”**

- 唯有在概率的意义上, “学习”才是可能的.
- 类比: bin of red and green marbles (用来揭示 probability sense of learning)
  - $\mu: P(\text{red}), or P(h(x) \neq f(x))$
  - $\nu$ : 样本中红球比例, or 样本中错误率
  - bin 对应一个概率分布确定 (或 h 确定) 的 input space.  hypothesis set H 对应 multiple bins.

![img](https://d2mxuefqeaa7sj.cloudfront.net/s_FA9189E64E3F201143EF4ED387BEEA610619552CD2E7E2AA295C880EFD385019_1496195855797_image.png)

- the Hoeffding Inequality: $P[|\nu - \mu] \gt \epsilon] \leq 2e^{-2\epsilon^2N}$
  - 大数定理的一种形式
  - $\nu$ 是样本的随机变量 (统计量). $\mu$ 是 固定的 parameter, 是未知的 “真实值”
  - good news: 样本数 N 越大, $\nu$ 越逼近 $\mu$
  - bad news: 逼近的效果不是很好, 或者说, 不能对误差 $\epsilon$ 要求很严 ( $\epsilon^2$  significantly dampens N )
  - tradeoff: N, $\epsilon$, and probability bound.
- 逻辑上的转换 (cause vs. effect): $\nu \approx \mu \Rightarrow \mu \approx \nu$

**connection to learning**

- notation: in-sample/out-of-sample error: $E_{in}(h), E_{out}(h)$
- 对任意一个**固定**的 h, 有 $P[|E_{in}(h) - E_{out}(h)| > \epsilon] \leq 2e^{-2\epsilon^2N}$
  - this shows that verification (not learning) is feasible.
- Hoeffding 不等式成立的前提: `the hypothesis h is` `*fixed*` `before you generate the dataset D.`
- 而在学习过程中, 我们需要基于 D 从 H 中选择 g, 因此不满足这一前提. 
  - 参考 coin analogy 例子 (书中 Ex 1.10, 或 Hw2 Q1-2)
- 对于 final hypothesis $g$, 暂时只能得到:
  - $P[|E_{in}(g) - E_{out}(g)| > \epsilon] \leq 2Me^{-2\epsilon^2N}$
    - M 为 hypothesis 数. 越复杂的模型, M 越大, $E_{out}$ 与 $E_{in}$ 的差距就越大.
    - 常见的模型, 实际上 M 无穷大. 后面会解决这个问题.
  - 从 verification is feasible 到 learning is feasible

**notes**

- 统计学 和 机器学习 的区别之一:
  - 统计学中的理论较严谨, 会做较多或较强的假设
  - 机器学习理论一般更注重实际中的普遍适用性, 减少所需的前提假设


## Lec3 - The Linear Model I

**input representation**

- 特征提取: 仅仅是简单地提取出两个特征( intensity, symmetry), 在两个手写数字的二分类问题上就能有比较好的效果

**Linear classification**

- PLA 用于线性不可分数据时会有震荡
- modification to PLA: the “pocket” algorithm

![img](https://d2mxuefqeaa7sj.cloudfront.net/s_FA9189E64E3F201143EF4ED387BEEA610619552CD2E7E2AA295C880EFD385019_1496283384053_image.png)


**Linear regression**

- *以下的* $w, x, y$ *为向量, 严格写法应该是* $\mathbf{w, x, y}$*. 为书写方便, 不做区分.*
****- $E_{in}(w) = \frac{1}{N} \sum_{n=1}^N (w^Tx_n - y_n)^2 = \frac{1}{N}\|Xw - y\|^2$


  - $X =\left[ \begin{array}{cc} - x_1^T - \\ -x_2^T- \\ ... \\ -x_N^T-\end{array} \right]$, $y = \left[ \begin{array}{cc} y_1 \\ y_2 \\ ... \\ y_N \end{array} \right]$


- $\nabla E_{in}(w) = \frac{2}{N} X^T(Xw - y)$. 令 $\nabla E_{in}(w) = 0$ , 则 $X^TXw = X^Ty \Rightarrow w = X^{\dagger}y$ 
  - $X^{\dagger} = (X^TX)^{-1}X^T$(pseudo-inverse of X)
  - X is a tall matrix. 一般 $X^TX$ 是可逆的.
- // 这种 Normal Equation 方法的复杂度是 $O(n^3)$ , 当 n 很大时, 效率不如梯度下降法 ($O(kn^2)$)
- // $\frac{\partial(Xw)}{\partial w} = X^T$

**nonlinear transformation**

- 对线性模型, 如线性回归 $\sum_{i=0}^dw_ix_i$ 或感知机 $sign(\sum_{i=0}^dw_ix_i)$, **“线性” 是关于权重** $w_i$ **, 而不是关于输入** $x_i$ **.** (linearity in weights, not in inputs/data**) .** 
  - 因此, 对数据做 non-linear transformation 不影响模型的线性本质
- transform the data nonlinearly to get new features
  - features are high level representations of raw inputs
- nonlinear transformation
  - $\mathbf{x \rightarrow z}$
  - $g(\mathbf{x}) = sign(\tilde \mathbf{w} \Phi(\mathbf{x}))$.  权重作用于变换后的空间 $\mathcal{Z}$ 而非原始输入空间 $\mathcal{X}$
  - transformation 若选择不当, 会造成泛化的困难
- 例子: Hw2 - Q8-10. 给定”真实”决策界面为圆形的数据.
  - transformation: $(1, x_1, x_2) \rightarrow (1, x_1, x_2, x_1x_2, x_1^2, x_2^2)$
## Lec4 Error and Noise

**Error measures**

- **也叫 error function / cost / objective / risk**
- to quantify how well h approximates f
- $E_{in}(h) = 1/N \sum_{n=1}^N e(h(x_n), f(x_n))$
- $E_{out}(h) = \mathbb{E}_x [e(h(x), f(x)]$ (期望)
- 理想情况下, **error measure should be user-specified**
  - 如二分类问题, TN 和 FP 的权重随场景而不同
- 但实际中, 难得有这种理想的 error measure. 原因一是用户本身无法提供, 二是 weighted cost 可能不容易优化. 这时可以按 analytic 或 practical 的原则来选择:
  - Plausible measures (analytically good) : 如 squared error $\equiv$ Gaussian noise
  - Friendly measures (easy to use/optimize) : 如 closed-form solution, convex optimization

**Noisy targets**

- 学习的目标一般情况下并不是 deterministic function, 而是一个概率分布 $P(y|x)$.
- noisy target = [deterministic target $f(x) = E(y|x)$] + [noise $y-f(x)$]
- 考虑噪音, 则 $y=f(x)$ → $y \sim P(y|x)$ , $E_{out}(h)$ → $\mathbb{E}_{x,y} [e(h(x), y]$
- 加入 error measure 和 noisy targets 后的 learning diagram:

![img](https://d2mxuefqeaa7sj.cloudfront.net/s_FA9189E64E3F201143EF4ED387BEEA610619552CD2E7E2AA295C880EFD385019_1496375358080_image.png)

- $P(x)$ vs $P(y|x)$
  - $P(x)$ 是一个未知的概率分布. 我们假定 x 服从某种分布, 使 Hoeffding 不等式成立, 从而使学习成为可能. 但我们不需要知道 P(x) 究竟是什么.
  - $P(y|x)$ 是 unknown target distribution, 是学习的目标 (注: 不是 $P(x, y)$ ).
  - 训练数据由 $P(x,y)$ 独立同分布生成.

**接下来 12 个 lecture 要下一盘很大的棋**

- 我们需要的结果: $E_{out}(g) \approx 0$ . 可拆分为:
  - $E_{out}(g) \approx E_{in}(g)$
  - $E_{in}(g) \approx 0$
- 因此, **本课程的两个主要问题**
  - **能否使** $E_{out}(g)$ **逼近** $E_{in}(g)$ **?  ← Lec 5~8**
  - **能否使** $E_{in}(g)$ **足够小? ← Lec 9-16**


## Lec5 Training versus Testing

**解决 M 无穷大的问题**

- $P[|E_{in} - E_{out}| > \epsilon] \leq 2Me^{-2\epsilon^2N}$
  - M 是为利用/污染了训练数据而付出的代价
  - M 怎么来的: 概率取了 union bound, 忽略了不同事件的重叠部分.
    - $P[B_1 or B_2 or ... or B_M] \leq P[B_1] + P[B_2] + ... + P[B_M]$  (B for Bad thing)
  - 需要从 H 中抽象出一个能考虑 overlap 的量 → break point
- dichotomies  (mini-hypotheses)
  - a hypothesis: $h: \mathcal{X} \rightarrow \{-1,+1\}$. 数量可以是无穷大.
  - a dichotomy: $h: \{x_1,x_2,...,x_N\} \rightarrow \{-1,+1\}$. 数量有限.
  - 以2D感知机为例: 
    - hypothesis 可以是任意直线. 
    - dichotomy 跟数据量 N 有关, N = 2, 3, 4 时, 数量上限分别为 4, 8, 14.
  - h 在整个 input space 上的结果 缩小至 h 在 N 个数据上的结果
  - 可看作**有效的** hypothesis
- the growth function 
  - $m_{\mathcal{H}}(N) = max_{x_1,...,x_N\in X}|\mathcal{H}(x_1,...,x_N)| \leq 2^N$
  - 表示 dichotomies 数量随 N 的增长. 可衡量 假设空间H 的复杂程度或”拟合能力”
  - $|\mathcal{H}(x_1,...,x_N)|$ 跟数据点的选取有关, 所以要取最大值.
  - 可看作 **effective** number of hypotheses

**examples (growth functions)**

- positive rays: linear growth function
- positive intevals: quadratic
- convex sets: $2^N$ (N points are “shattered”)

![img](https://d2mxuefqeaa7sj.cloudfront.net/s_FA9189E64E3F201143EF4ED387BEEA610619552CD2E7E2AA295C880EFD385019_1496474799988_image.png)


**break point**

- 定义
  - 使 H 无法 shatter 数据集的 k  / 使 $m_{\mathcal{H}}(k) < 2^k$ 的 k
  - 定义中没有限定 “最小”, 即: 如果 k 是 break point, k+1 也是 break point. 但后面的讨论经常指的是 smallest break point.
- 例子: positive rays, positive intervals, 2d perceptron 的 break point
- 只要存在 break point, 增长函数就是多项式的. 意味着 learning is feasible.
- break point 的数值有助于决定某个问题需要多少数据.
- 例子: 若 k = 2, 则 N = 3 时的 dichotomies 数量会减少一半.

![img](https://d2mxuefqeaa7sj.cloudfront.net/s_FA9189E64E3F201143EF4ED387BEEA610619552CD2E7E2AA295C880EFD385019_1496628427544_image.png)

## Lec6 Theory of generalization

**Proof that** $m_H(N)$ **is polynomial**

- $B(N,k)$ : maximum number of dichotomies on N points, with break point k. 不限定 H, 故 $m_H(N) \leq B(N,k)$.
- 用递归的方法找 $B(N,k)$ 的上界: $B(N,k) \leq \sum_{i=0}^{k-1} {N \choose i}$
- 所以 $m_H(N) \leq \sum_{i=0}^{k-1} {N \choose i}$ , max power is $N^{k-1}$

**Proof that** $m_H(N)$ **can replace M**

- 结合 **Textbook 2.1.4 The VC generalization bound**
- 证明过程较复杂, 忽略.
- **The Vapnik-Chervonenkis Inequality**
  - $P[|E_{in}(g) - E_{out}(g)| > \epsilon] \leq 4m_H(2N) e^{-e^2N/8}$


## Lec7 The VC dimension

**The VC dimension**

- The VC dimension $d_{vc}$
  - the most points H can shatter
    - 只要有一组 N 个点能被 shatter 即可, 不需要 shatter any N points
  - 使 $m_H(N) = 2^N$ 的最大的 N
  - VC dimension vs. break point
    - $k = d_{vc} + 1$  is a break point (smallest)
    - break point is the failure to shatter, VC dimension is the ability to shatter.
  - $d_{vc}$ is the order of the polynomial bound on $m_H(N)$
- 可以证明 $\sum_{i=0}^D {N \choose i} \leq N^D + 1$, 因此 $m_H(N) \leq N^{d_vc} + 1$
- finite $d_{vc}$ $\Rightarrow$ good models, $g\in H$ will generalize.
  - 仅与 H 和 D 有关. 不依赖于算法/input distribution/target function. 因而 VC analysis 具有较好的一般性.
  - diversity 的极端: $m_H(N) = 2^N, d_{vc}(H) = \infty$. 此时模型完全没有泛化能力.
- 例子: d 维感知机的 $d_{vc} = d + 1$, 刚好等于参数个数
- 复杂模型如神经网络的 d_vc 很难准确获得, 只能得到一个 loose bound, 用于相对比较.

**Interpreting**

- **VC dimension 与模型自由度的关系**
  - VC dimension 可看做**”有效”**的参数个数, 或者模型的”自由度’. (The VC dimension measures these effective parameters or 'degrees of freedom' that enable the model to express a diverse set of hypotheses.)
  - parameters may not contribute DOF, 比如串联起来的感知机.
  - VC dimension 是对自由度更为可靠的衡量.
- **VC dimension 与 需要的数据量 的关系**
  - rule of thumb: **to get reasonable generalization,** $N \geq 10 d_{vc}$

**Generalization bounds**

- $E_{out} \leq E_{in} + \Omega(N,H,\delta)$ with probability $\geq 1-\delta$
  - $\Omega(N,H,\delta) = \sqrt{\frac{8}{N} ln\frac{4m_H(2N)}{\delta}}$ : bound to generalization error.
- 当假设空间扩大时,  $E_{in}$ 减小, 而 $\Omega$ 增大. 为了最优化 $E_{out}$ , 需要这两项之间的某种平衡.
- 另外这个式子与 正则化 有关. → Lec12


## Lec8 Bias-Variance tradeoff

lec 5-8 都是关于 generalization 问题, 或者说 approximation-generalization tradeoff

- Lec5~7 以二分类的 binary error 为例, 分析方法是 VC analysis, 结果是 $E_{out} \leq E_{in} + \Omega$
- Lec8 以 squared error measures 为例, 分析方法是 bias-variance analysis. 结果是将 $E_{out}$ 分解为 bias + variance.

**Bias and Variance**

- **bias-variance 分析的对象是** $\mathbb{E}_D(E_{out}(g^D))$ **, 对 D 和 x 都取了期望, 反映的是不依赖于特定 D 且基于整个输入空间的规律.**
  - 其中 $E_{out}(g^D) = \mathbb{E}_x[(g^D(x) - f(x))^2]$
  - 课程前面提到的 $E_{out}$ 实际上都跟 D 有关, 只是为表达的简洁而略去了.
- a conceptual tool: **average function** $\overline{g}(x) = \mathbb{E}_D[g^D(x)]$
  - 可理解为: 随意选择 D 能得到的最好的 g. 
  - $\overline{g}$ 与 f 的差距反映了 H 的局限性.
- 分解: $\mathbb{E}_D(E_{out}(g^D)) = bias + var$
  - squared error 保证了可以这样分解而没有交叉项.
  - tradeoff: 当 H 扩大时, bias 减小而 variance 增大.
  - Lec11 进一步增加了一个 noise 项

![img](https://d2mxuefqeaa7sj.cloudfront.net/s_FA9189E64E3F201143EF4ED387BEEA610619552CD2E7E2AA295C880EFD385019_1497067459235_image.png)

- example: 给定*两个*数据点, 分别用 h(x) = b 和 h(x) = ax + b 去学习 f(x) = sin(pi*x)
  - 结果1个参数的前者表现更好. 可见模型复杂度需要与数据量匹配.

![img](https://d2mxuefqeaa7sj.cloudfront.net/s_FA9189E64E3F201143EF4ED387BEEA610619552CD2E7E2AA295C880EFD385019_1497069851514_image.png)


**Learning Curve**

- simple model vs complex model

![img](https://d2mxuefqeaa7sj.cloudfront.net/s_FA9189E64E3F201143EF4ED387BEEA610619552CD2E7E2AA295C880EFD385019_1497081784646_image.png)

  - 纵轴是 E 关于 D 的期望: 消除 D 的不同选择带来的随机性, 反映更一般的规律.
- 在 learning curve 上分别解释 VC analysis 和 bias-variance
  - **中间的 bias 水平线代表 best approximation in H**, 对应 $\overline{g}(x)$ . 这里其实存在一个理想化的假定, 即 best approximation (或者说 bias) 与 N 无关.

![img](https://d2mxuefqeaa7sj.cloudfront.net/s_FA9189E64E3F201143EF4ED387BEEA610619552CD2E7E2AA295C880EFD385019_1497082123229_image.png)

- learning curves for linear regression
  - 这个例子中, 可以求得曲线方程, E_in 与 E_out 对称
  - generalization error = $2\sigma^2(\frac{d+1}{N})$ , $\sigma$ 是 noise 的标准差
    - to get reasonable generalization, $N \varpropto d_{vc}$ . 这与分类问题的结论一致.


## Lec9 The linear model II
> The linear model can be used as a building block for other popular tech­niques. A cascade of linear models, mostly with soft thresholds, creates a neural network. A robust algorithm for linear models, based on quadratic programming, creates support vector machines. An efficient approach to non­linear transformation in support vector machines creates kernel methods. A combination of different models in a principled way creates boosting and en­semble learning. There are other successful models and techniques, and more to come for sure.
> -- Learning From Data (book), P181

**generalization in nonlinear transformation**

- 变换前: $\mathbf{x} = (x_0,x_1,...,x_d)$ , $d_{vc} = d + 1$
- 变换后: $\mathbf{z} = (z_1,z_2,...,z_{\tilde d})$ ,  $d_{vc} \leq \tilde d + 1$
- 付出的代价是 d_vc 会增大
- 两个例子:
  - 例1-左图: approximation-generalization tradeoff. 为了把错误率降至0而强行使用非线性变换, 会严重降低泛化能力. 相比之下, 线性分界面的错误率是可以接受的.
  - 例2-右图: 从 $(1,x_1,x_2)$ 变换到 $(1,x_1,x_2,x_1x_2,x_1^2,x_2^2)$. d_vc 从 3 增大到 6, 需要的数据量大致翻倍.
    - Q: 若用 $(1,x_1^2,x_2^2)$ 甚至 $(1,x_1^2 + x_2^2)$ , 降低了 d_vc, 是不是好的选择?
    - A: 选这样的特征, 是以预先偷看数据为代价的, 会影响实际的泛化能力. 这是 `data snooping`

![img](https://d2mxuefqeaa7sj.cloudfront.net/s_FA9189E64E3F201143EF4ED387BEEA610619552CD2E7E2AA295C880EFD385019_1497245036842_image.png)


**logistic regression**

- 相比 sign 函数的 hard threshold, logistic function 提供了 soft threshold (reflects uncertainty). 
- probability interpretation
  - target function: $f(x) = P[y=+1|x]$
  - the data does not give us the value of f explicitly. Rather, it gives us samples generated by this probability.
  - therefore, the data is generated by a noisy target $P(y|x)$
    - $P(y|x) = \left\{ \begin{array}{lcc} f(x) & \text{for} & y = +1; \\ 1-f(x) & \text{for} & y = -1. \end{array}\right.$
  - *注意这里 y 取为 {+1, -1}. 其他书上多为 {0, 1}.* 
- error measure
  - ​​`likelihood` : outcome 已知, 参数未知.
  - log likelihood: $l(w) = \sum_{i=1}^N\ln p(y_i|x_i; w)$, 其中 $p(y_i|x_i) = \theta(y_iw^Tx_i)$
  - $E_{in}(w) = \frac{1}{N} \sum_{n=1}^N \ln(1+e^{-y_nw^Tx_n})$  → **cross-entropy error**
- algorithm: gradient descent. 忽略高阶项.
  - **learning rate = (step size) / (norm of gradient)**

**misc**

- 为什么不用二阶优化方法: 单步计算代价增大, 得不偿失.
- conjugate gradient?
- 区分: 根据对问题的理解选择特征 (good) vs. 根据数据选择特征 (bad)


## Lec10 Neural Networks

**SGD**

- SGD 带来的随机性有助于避开局部最小值
- rule of thumb: 学习率取 0.1

**neural network model**

- 多层感知机的优化很困难. 神经网络则把其中的 hard threshold 换成 soft threshold, 平滑可微, 更易求解.
- notations
  - $w_{ij}^{(l)} \left\{ \begin{array}{ll} 1\leq l\leq L & \text{layers} \\ 0\leq i\leq d^{(l-1)} & \text{inputs} \\ 1\leq j\leq d^{(l)} & \text{outputs} \end{array} \right.$
  - $\theta(s) = \tanh(s) = \frac{e^s - e^{-s}}{e^s + e^{-s}}$
    - **s ~ signal****,** $\theta$ **~ threshold**
  - $x_j^{(l)} = \theta(s_j^{(l)}) = \theta( \sum_{i=0}^{d^{(l-1)}} w_{ij}^{(l)} x_i^{(l-1)})$

![img](https://d2mxuefqeaa7sj.cloudfront.net/s_FA9189E64E3F201143EF4ED387BEEA610619552CD2E7E2AA295C880EFD385019_1497683136938_image.png)


**BP algorithm**

- 讲 BP 时顺带提到了 FFT. 这两个算法的优点不在于能解决问题, 而在于高效地解决问题.
- 实施 SGD
  - 单个样本的 cost: $e(h(x_n), y_n) = e(\mathbf{w})$ , $\mathbf{w} = \{ w_{ij}^{(l)} \}$
  - 为了实施 SGD, 我们需要梯度 $\nabla e(w): \frac{\partial e(w)}{\partial w_{ij}^{(l)}} \text{ for all } i,j,l$
- $\frac{\partial e(w)}{\partial w_{ij}^{(l)}}$ 的计算
  - 笨办法: 挨个计算, analytically or numerically
  - a trick for efficient computation
    - cost 对权重的偏导可拆成 x 项和 delta 项相乘, delta 是 cost 对 signal 的偏导
      - 图解: 权重被上一层的 x 和下一层的delta 操控, 夹在中间
    - x 项可由前向传播计算
    - 重点是 delta 项, 用反向传播

![img](https://d2mxuefqeaa7sj.cloudfront.net/s_FA9189E64E3F201143EF4ED387BEEA610619552CD2E7E2AA295C880EFD385019_1497686399906_image.png)

![img](https://d2mxuefqeaa7sj.cloudfront.net/s_FA9189E64E3F201143EF4ED387BEEA610619552CD2E7E2AA295C880EFD385019_1497689277026_image.png)

- 反向传播计算 $\delta$
  - 注: 对于 tanh, $\theta'(s) = 1 - \theta^2(s)$

![img](https://d2mxuefqeaa7sj.cloudfront.net/s_FA9189E64E3F201143EF4ED387BEEA610619552CD2E7E2AA295C880EFD385019_1497687922530_image.png)

- BP 算法
  - 初始化权重 $w_{ij}^{(l)}$ 
    - 随机值, 不能为0. 
  - 循环
    - pick $n\in \{1,2,...,N\}$
    - 前向: 计算各层 $x_j^{(l)}$
    - 反向: 计算各层 $\delta_j^{(l)}$
    - 更新 weights: $w_{ij}^{(l)} \leftarrow w_{ij}^{(l)} - \eta x_i^{(l-1)} \delta_j^{(l)}$
  - 得到 final weights
- hidden layer 可以看做 **“learned nonlinear transform”**
  - 可以理解为, 神经网络自己做了 data snooping, 但同时也付出了相应的代价 (参数数量和 d_vc 增加)
  - data snooping (或者说 looking at the data) 并不是说一定不行. 而是, 通过 data snooping 得到的模型, 它的实际泛化能力会比看起来要差. 如果你能为此付出相应的代价, 那 data snooping 也不是不可以. 但问题是, 人为的 data snooping, 你很难准确衡量这种代价.


## Lec11 Overfitting
- lec 11-13 都是围绕这一主题. 对付过拟合的两大武器: regularization + validation
- 这里提到了 target function 的复杂度和 target distribution 的噪音, 需要注意, 实际中这些关于 target 的信息都是无从获得的. 这种分析相当于站在上帝视角带着你理解一些概念.

**what is overfitting**

- overfitting: 指 fitting the data (即降低 E_in) 已无助于改进 E_out.
  - 典型场景: 较复杂的模型把多余的自由度拿来学习噪音了.
  - 例1: 多项式回归中使用不必要的高次函数
  - 例2: 在一个神经网络训练过程中, epoch 数超过某个值时, E_out 反而上升. (也可以叫 over-training, 对策是 early stopping)
  - **overfitting vs. bad generalization**
    - overfitting 经常伴随泛化能力下降. 但有时不然, 比如例2中的 overfitting 发生在同一个模型里(或者说发生在一个”过程”里), 不涉及模型选择和泛化能力.
- 元凶: fitting the noise (stochastic/deterministic)

**the role of noise**

- 两个例子:
  - 不同的 target, 15个数据点, 分别用2阶和10阶多项式去拟合. 结果都是2阶的 E_out 表现更好.
  - 例1: target 为10阶多项式, 数据有 noise
    - target 是10阶函数不代表适合用10阶的假设函数去拟合. 因为噪音的存在, 假设函数的选择首先应该匹配数据量(包括数量和质量), 而不是去匹配目标函数的复杂度.
  - 例2: target 为50阶多项式, 数据无 noise
    - 这个例子里 target 的复杂度造成了一种新的 noise.
    - 即便 hypothesis 比 target function 简单很多, 依然会发生 overfit.

![img](https://d2mxuefqeaa7sj.cloudfront.net/s_FA9189E64E3F201143EF4ED387BEEA610619552CD2E7E2AA295C880EFD385019_1497697518797_image.png)

![img](https://d2mxuefqeaa7sj.cloudfront.net/s_FA9189E64E3F201143EF4ED387BEEA610619552CD2E7E2AA295C880EFD385019_1497698814579_image.png)

- 结合 learning curve: 灰色区域里, complex model 相比 simple model, bias 较小, 但 variance 很大; E_in 较小, E_out 较大.

![img](https://d2mxuefqeaa7sj.cloudfront.net/s_FA9189E64E3F201143EF4ED387BEEA610619552CD2E7E2AA295C880EFD385019_1497699493281_image.png)

- 一个实验: 研究 noise level 和 target complexity 对过拟合的影响
  - stochastic noise: 即之前一直在讲的 noise, 等于 y - f(x)
  - deterministic noise: 源于 target complexity 的一种更宽泛的 “noise”

![img](https://d2mxuefqeaa7sj.cloudfront.net/s_FA9189E64E3F201143EF4ED387BEEA610619552CD2E7E2AA295C880EFD385019_1497698536296_image.png)

![img](https://d2mxuefqeaa7sj.cloudfront.net/s_FA9189E64E3F201143EF4ED387BEEA610619552CD2E7E2AA295C880EFD385019_1497698548158_image.png)


**deterministic noise**

- deterministic noise: the part of target $f$ that $\mathcal{H}$ cannot capture.
  - 超出模型(假设空间)的能力范围

![img](https://d2mxuefqeaa7sj.cloudfront.net/s_FA9189E64E3F201143EF4ED387BEEA610619552CD2E7E2AA295C880EFD385019_1497847013969_image.png)

- 区别于 stochastic noise 的特点: 依赖于 H; 对于给定 x 是确定的.
  - 实际中的机器学习问题, H 和 D 一般是确定的, 因此两种 noise 看不出区别.

**bias-variance-noise 分解**

  - 对 $\mathbb{E}_D(E_{out})$ 的分解可新增一项 stochastic noise. 而 bias 其实就是 deterministic noise.

![img](https://d2mxuefqeaa7sj.cloudfront.net/s_FA9189E64E3F201143EF4ED387BEEA610619552CD2E7E2AA295C880EFD385019_1497708067874_image.png)

  - 对这三项的理解
    - variance 取决于 D 与 H 复杂性 (或信息量?) 的相对关系, D 越小而 H 越复杂时, variance 就越大.
    - bias 反映了 H 能力的局限性. **与 D 无关**.
    - stochastic noise 取决于 target distribution P(y|x). **与 D 和 H 均无关**.
  - 借助 bias-variance-noise 分解, 理解 **how noise affects performance** (书 P125)
    - bias 和 $\sigma^2$ 项反映了两种 noise 对 cost 的**直接**影响
      - 只要 H 确定, 不管是 overfit 还是 underfit 情况, 这两项 cost 都是恒定的, 与D无关.
      - underfit 情况下 bias 可能在整个 cost 中占主导.
    - variance 项则包含了 noise 对 cost 的**间接**影响. (noise 误导模型跑偏了)
      - **The var term is indirectly impacted by both types of noise, capturing a model’s susceptibility to being led astray by the noise.**
      - **overfit 情况下这一项主导**. 反之, 当 N 很大时，var 趋于0，也就不存在 overfit.
      - noise 越大, overfit 的可能性越大. 极端情况下, 如果数据是纯粹的 noise, 那么无论怎么拟合都是过拟合.
    - 所谓 “noise 引起 overfitting”, 可以更准确地表达为: 当 H 有多余自由度时, noise 会诱发模型跑偏, 增大 variance, 发生过拟合. (当 H 足够简单时, 模型都不具有”跑偏”的能力)
  - 注意1: 分解的对象是对 D 和 x 取了期望的 E_out. 分解结果中, 只有第一项 variance 还与 D 有关; 而三项都包含 对 x 的期望, 反映在输入空间上的总体规律, 而非针对具体数据点.
  - 注意2: 这个分解也是上帝视角. 面对实际问题时, 模型无法区分 signal 和 noise, 只要有多余的自由度, 就会不自觉地去拟合 noise.
  - overfitting 与 bias-variance tradeoff
    - 选择较复杂的 H 可以降低 bias (deterministic noise), 因为 H 的复杂度与 target 更接近了.
    - 但在数据量不足时, 复杂模型会用多余的自由度会去学习噪音, 结果造成 variance 很大. 而 variance 的增大很容易就会抵消 bias 的减少, 因而总的 cost 会增大, 也就是发生了过拟合.

**小结**

- overfitting 是指降低 E_in 无助于降低 E_out. 诱因是 noise. 发生的条件是模型 capacity 富余或数据不足.
- stochastic/deterministic noise. 前者与H无关, 后者与H有关. 但实践中两种 noise 表现相似.
- bias-variance-noise 分解. bias 和 noise 反映了 noise 的直接影响, variance 包含了 noise 的间接影响.


## Lec12 Regularization

**regularization**

- mathematical approach and heuristic approach
  - 数学方法所需的假定, 实际中经常难以满足. 但从中得到的 intuition 可用于指导 heuristic 方法.
  - “regularization is as much an art as it is a science.” 实际中使用的大多为 heuristic methods.
- regularization 为”拟合”过程增加了约束, 会同时限制模型对 signal 和 noise 的拟合. 因此一方面增加 bias, 一方面减少 variance.
- regularization 是一种 soft constraint. 与之相对, 直接把某些函数排除在 H 之外, 是一种 hard constrain.
- 数学推导
  - 使用 Legendre polynomials. 特点: 彼此正交, 参数独立.
  - 有约束的优化问题: minimize $E_{in}(w)$, 满足 $w^Tw \leq C$  ← soft-order constraints
  - 等价于一个无约束的优化问题: minimize $E_{aug}(w) = E_{in}(w) + \frac{\lambda}{N}w^Tw$ ← augmented error
    - C 越小, $\lambda$ 越大 ( $\lambda$ 还与其他因素有关, 无法解析地用 C 表示)
  - 解得: $w_{reg} = (Z^TZ + \lambda I)^{-1}Z^Ty$
- **regularizer 的选择一般用 heuristic 方法.** $\lambda$ **的选择则基于 validation.**
- noise 越多, 需要的正则化系数越大

**weight decay**

- 对 $E_{in}(w) + \frac{\lambda}{N}w^Tw$ 的最小化, 就叫 weight decay. 这是最常见的一种 regularizer.
- 为啥叫 weight decay? 在梯度下降求解时:

![img](https://d2mxuefqeaa7sj.cloudfront.net/s_FA9189E64E3F201143EF4ED387BEEA610619552CD2E7E2AA295C880EFD385019_1497922398321_image.png)

- 在 w 更新过程中, 两股力量在起作用: 除了 最小化 E_in, 另一种力量是 w 逐渐减小. 后者使 w 不能按自由地沿 E_in 梯度方向变化.
- variations of weight decay
  - emphasis of certain weights  $\sum_{q=0}^Q \gamma_q w_q^2$  ← diagnal quadratic form
    - 比如, 可以设置不同的 $\gamma_q$ 使模型偏向 low-order fit 或 high-order fit.
  - 神经网络: 不同层用不同 $\gamma$
  - a general form: Tikhonov regularizer $w^T\Gamma^T\Gamma w$  ← general quadratic form
- practical rule: 
  - stochastic noise 是”高频”的
  - deterministic noise 是不平滑的
  - it helps to constraint learning toward smoother hypotheses. ← weight decay 类型的正则化可以做到这一点.
- general form of augmented error
  - $E_{aug}(h) = E_{in}(h) + \frac{\lambda}{N} \Omega(h)$ . $\Omega$ ~ regularizer.
    - related to generalization bound: $E_{out}(h) \leq E_{in}(h) + \Omega(H)$
    - $\Omega(H)$ 与 $\Omega(h)$ 的联系见 Lec17.
  - E_aug is better than E_in as a proxy for E_out

**choosing a regularizer**

- guiding principle: direction of **smoother or simpler**
  - 比如 netflix 用户评分预测的例子: 采用了使预测趋于平均值的正则化
- neural-network regularizers
  - weight decay
    - 由于 tanh 的性质, 当参数很小时, 趋于线性; 当参数很大时, 非线性显著.
  - weight elimination
    - 参数越少, d_vc 越小
    - 一种 soft weight elimination: $\Omega(w) = \sum_{i,j,l} \frac{(w_{ij}^{(l)})^2} {\beta^2 + (w_{ij}^{(l)})^2}$
      - 小的参数更趋于 0 (softly eliminated), 大的参数不受影响.
- early stopping 也是一种 regularizer

**misc**

- 区分三种 regularizer
  - weight decay 中使模型偏向 low-order fit 或 high-order fit
  - weight decay 的反面: weight growth (不可取)
  - soft weight elimination: softly eliminate small weights
- less features vs. more features + regularization: 后者更为灵活强大
- 小结: 
  - regularization 是一种 soft constraint. 参数受约束时 E_in 的优化问题 等价于无约束的 augmented error 优化问题. 后者是正则化的一般形式.
  - 常用的 regularizer: weight decay. 使函数趋于平滑.


## Lec13 Validation

**The validation set**

- validation vs. regularization
  - 都在试图优化 E_out. E_out = E_in + overfit penalty
  - regularization estimates “overfit penalty”
  - validation estimates E_out
- K 个点的 validation set: 
  - $\mathbb{E}[E_{val}(h)] = E_{out}(h)$
    - E_val(h) 是对 E_out(h) 的无偏估计. 
    - 注: h 是任一假设函数, 未依据 validation 做模型比选
  - $\mathrm{var}[E_{val}(h)] = \sigma^2/K$
  - $E_{val}(h) = E_{out}(h) \pm O(\frac{1}{\sqrt{K}})$
- K 个点用完后还可以放回训练数据
  - 为什么可以放回? 因为放回前已经完成了 validation, 得到了需要的 E_val, 这时放回不会影响什么.

![img](https://d2mxuefqeaa7sj.cloudfront.net/s_FA9189E64E3F201143EF4ED387BEEA610619552CD2E7E2AA295C880EFD385019_1498106344132_image.png)

- rule of thumb: K = N/5

**Model selection**

- “validation” 的含义: 利用这部分数据来做选择
- 做了选择之后, E_val 不再是 E_out 的无偏估计, 会有一个 optimistic bias

![img](https://d2mxuefqeaa7sj.cloudfront.net/s_FA9189E64E3F201143EF4ED387BEEA610619552CD2E7E2AA295C880EFD385019_1498107112508_image.png)

- validation 可以看做是对一个特殊的假设空间 H_val 的”训练”
  - 不过 这个”训练”的 “强度” 不大, 保证 E_val 相对于 E_out 的 bias 不会很大.
  - $H_{val} = \{ g_1^-, g_2^-,...,g_M^- \}$, 即 validation 备选的 M 个模型分别训练得到的 final hypothesis.
  - 可以用 VC generalization bound, 得到 E_val 的 bias 的上限

![img](https://d2mxuefqeaa7sj.cloudfront.net/s_FA9189E64E3F201143EF4ED387BEEA610619552CD2E7E2AA295C880EFD385019_1498114450497_image.png)

- data contamination
  - 三种 error **estimates** (of E_out): E_in, E_val, E_test
  - 数据污染的程度, 也就是这三种 estimate 的 optimistic bias 的大小.
  - training set: 完全污染
  - validation set: 轻微污染
    - 为了控制污染程度, 当需要 validate 的超参数较多时, 可以设置多个 validation set
    - validation set size 和 hyper-parameter 数量的合理比例: 建议是大约 100 data points ~ a couple of hyper-parameters
  - test set: 完全纯净
- validation 作为一种选择模型的方法, 优点是基本不需要什么假定.

**Cross validation**

- the dilemma about K
  - 用 E_val 准确估计 E_out, 需要做好下图的两个环节
  - K 过大时, 依靠 N-K 个训练数据得到的 $g^-$ 不理想
  - K 过小时, E_val 的估计不准 (variance 很大)  ←  cross validation 可解决此问题

![img](https://d2mxuefqeaa7sj.cloudfront.net/s_FA9189E64E3F201143EF4ED387BEEA610619552CD2E7E2AA295C880EFD385019_1498123702917_image.png)

- 一种极端情况: leave one out
  - K = 1,  $g^-$最接近 g
  - cross validation error 定义为 $E_{cv} = \frac{1}{N} \sum_{n=1}^Ne_n$ , 其中 $e_n$ 是每种 train/val set 划分方式下的 error
  - N 个独立同分布的随机变量求平均后的 variance 会降到 1/N.
  - 显然这 N 个 $e_n$ 并不独立, 因为它们用到的数据有很大重叠. 但如果分析 $E_{cv}$ 的 variance, 会发现它的 “有效” 数据量接近 N (而不是 1) . 也就是说, 这些 e_n 接近于相互独立.
    - 这是 cross-validation 魔法的来源
  - 一个例子:

![img](https://d2mxuefqeaa7sj.cloudfront.net/s_FA9189E64E3F201143EF4ED387BEEA610619552CD2E7E2AA295C880EFD385019_1498117888904_image.png)

- 但在实际中
  - leave one out 不现实, 意味着需要进行 N 组训练. 
  - 可改用 V fold cross validation. 常见的是 V = 10, 即 K = N/10.


## Lec14 SVM

**Maximizing the margin**

- 要求 fat margin 意味着假设空间缩小, d_vc 变小
- 点 x_n 到超平面 $w^Tx + b = 0$ 的距离
  - 先 normalize w, 使 $|w^Tx_n + b| = 1$
  - 距离 = 1/||w||
- “间隔最大” 等于以下优化问题:
  - 最大化 1/||w|| , 满足 $min_n |w^Tx_n + b| = 1$
- 转化为更易求解的形式
  - 最小化 $w^Tw/2$ , 满足 $y_n(w^Tx_n + b) \geq 1$ for n=1,2,…N

**The solution**

- 求解比较复杂. 需要用到 QP 的知识.
  - "quadratic programming”: 受线性约束的二次优化问题求解
- 问题与前面的 regularization 类似. 有趣的是后者是约束 $w^Tw$ 优化 E_in, 而 SVM 刚好反过来.
- 优化目标:

![img](https://d2mxuefqeaa7sj.cloudfront.net/s_EB6E580B0B6ABFCDC87F08E8E963535A27D1C04D2C2D29F372803D92C91B7248_1498191119604_image.png)

- 对 w 和 b 求梯度. 令梯度为0, 可以将 w 和 b 用 $\alpha$ 表示. 代入 $L(w,b,\alpha)$ 得到 $L(\alpha)$.

![img](https://d2mxuefqeaa7sj.cloudfront.net/s_EB6E580B0B6ABFCDC87F08E8E963535A27D1C04D2C2D29F372803D92C91B7248_1498193699863_image.png)

- 优化 $L(\alpha)$ : 把以下设定传给 QP 模块

![img](https://d2mxuefqeaa7sj.cloudfront.net/s_EB6E580B0B6ABFCDC87F08E8E963535A27D1C04D2C2D29F372803D92C91B7248_1498189537551_image.png)

- “support vector” 的含义: 即刚好在间隔边缘的点.

![img](https://d2mxuefqeaa7sj.cloudfront.net/s_EB6E580B0B6ABFCDC87F08E8E963535A27D1C04D2C2D29F372803D92C91B7248_1498191678804_image.png)

- $\alpha_n$ 可以看做是模型的参数. 因为仅对 support vector 才有 $\alpha_n > 0$, 大部分 $\alpha_n = 0$. 所以 $w = \sum_{x_n\text{ is SV}} a_ny_nx_n$
  - “有效”参数的个数大大减少了. 有助于泛化能力.

**Nonlinear transforms**


![img](https://d2mxuefqeaa7sj.cloudfront.net/s_EB6E580B0B6ABFCDC87F08E8E963535A27D1C04D2C2D29F372803D92C91B7248_1498192408312_image.png)

- 从 输入空间 X 做非线性变换到 Z, 只需要把 $L(\alpha)$ 里的 x 内积换为 z 内积. **即使 Z 空间维数增大很多, 计算量并不会增加多少.**
- “support vector” 图解
  - 左图为一个线性 SVM 的 margin 和 support vector. 也可以看做一个非线性 SVM 在 Z 空间里的 margin 和 support vector.
  - 右图为 非线性 SVM 在 X 空间里的决策界面 和 support vector. 这条曲线其实就是 Z 空间里 margin 最大的直线.

![img](https://d2mxuefqeaa7sj.cloudfront.net/s_EB6E580B0B6ABFCDC87F08E8E963535A27D1C04D2C2D29F372803D92C91B7248_1498192760588_image.png)

![img](https://d2mxuefqeaa7sj.cloudfront.net/s_EB6E580B0B6ABFCDC87F08E8E963535A27D1C04D2C2D29F372803D92C91B7248_1498192768352_image.png)

- 泛化性能: $\mathbb{E}[E_{out}] \leq \frac {\mathbb{E} [\text{\# of SVs}] } {N-1}$ . **只要 SV 比例较低, 特征空间维度再高都不怕.**
  - 上式解读: 用一个 in-sample quantity 就可以检查 out-of-sample error
- 所以 SVM 厉害之处在于: 可以自由地在高维空间玩耍, 不必太担心计算量和泛化的问题.
  - **complex h, but (relatively) simple H**
- nonlinear transform 常与 soft margin 结合使用





