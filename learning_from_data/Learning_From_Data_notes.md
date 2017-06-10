# Learning_From_Data_lim
- Course materials: [Lectures](https://work.caltech.edu/lectures.html)  |  [Homeworks](https://work.caltech.edu/homeworks.html)
- 部分 Homework 解答和笔记: [homeworks.ipynb](https://github.com/sunoonlee/machine-learning/blob/master/learning_from_data/Homeworks.ipynb)

## Lec 1 - The Learning problem
https://d2mxuefqeaa7sj.cloudfront.net/s_FA9189E64E3F201143EF4ED387BEEA610619552CD2E7E2AA295C880EFD385019_1496033144892_image.png


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
  - $\mu: P(red), or P(h(x) \neq f(x))$
  - $\nu$ : 样本中红球比例, or 样本中错误率
  - bin 对应一个概率分布确定 (或 h 确定) 的 input space.  hypothesis set H 对应 multiple bins.
https://d2mxuefqeaa7sj.cloudfront.net/s_FA9189E64E3F201143EF4ED387BEEA610619552CD2E7E2AA295C880EFD385019_1496195855797_image.png

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
https://d2mxuefqeaa7sj.cloudfront.net/s_FA9189E64E3F201143EF4ED387BEEA610619552CD2E7E2AA295C880EFD385019_1496283384053_image.png


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
- 理想情况下, **error measure should be specified by the user.**
  - 如二分类问题, TN 和 FP 的权重随场景而不同
- 但实际中, 难得有这种理想的 error measure. 原因一是用户本身无法提供, 二是 weighted cost 可能不容易优化. 这时可以按 analytic 或 practical 的原则来选择:
  - Plausible measures (analytically good) : 如 squared error $\equiv$ Gaussian noise
  - Friendly measures (easy to use) : 如 closed-form solution, convex optimization

**Noisy targets**

- 学习的目标一般情况下并不是 deterministic function, 而是一个概率分布 $P(y|x)$.
- noisy target = [deterministic target $f(x) = E(y|x)$] + [noise $y-f(x)$]
- 考虑噪音, 则 $y=f(x)$ → $y \sim P(y|x)$ , $E_{out}(h)$ → $\mathbb{E}_{x,y} [e(h(x), y]$
- 加入 error measure 和 noisy targets 后的 learning diagram:
https://d2mxuefqeaa7sj.cloudfront.net/s_FA9189E64E3F201143EF4ED387BEEA610619552CD2E7E2AA295C880EFD385019_1496375358080_image.png

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
https://d2mxuefqeaa7sj.cloudfront.net/s_FA9189E64E3F201143EF4ED387BEEA610619552CD2E7E2AA295C880EFD385019_1496474799988_image.png


**break point**

- 定义
  - 使 H 无法 shatter 数据集的 k  / 使 $m_{\mathcal{H}}(k) < 2^k$ 的 k
  - 定义中没有限定 “最小”, 即: 如果 k 是 break point, k+1 也是 break point. 但后面的讨论经常指的是 smallest break point.
- 例子: positive rays, positive intervals, 2d perceptron 的 break point
- 只要存在 break point, 增长函数就是多项式的. 意味着 learning is feasible.
- break point 的数值有助于决定某个问题需要多少数据.
- 例子: 若 k = 2, 则 N = 3 时的 dichotomies 数量会减少一半.
https://d2mxuefqeaa7sj.cloudfront.net/s_FA9189E64E3F201143EF4ED387BEEA610619552CD2E7E2AA295C880EFD385019_1496628427544_image.png

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

