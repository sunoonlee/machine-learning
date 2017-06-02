Course materials: [Lectures](https://work.caltech.edu/lectures.html)  |  [Homeworks](https://work.caltech.edu/homeworks.html)

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
  - “using a set of observations to uncover an underlying process”
    - in statistics:
      - underlying process = probability distribution
      - observations = samples
- supervised learning: given (input, output)
- unsupervised learning: given only input
  - a way of getting a higher-level representation of the input
  - 听广播学外语
- reinforced learning: given (input, some output, grade for this output)

**misc**

- learning vs memorizing (overfitting)
- bottleneck of machine learning: generalization


## Lec2 - Is Learning Feasible?

**probability to the rescue 概率拯救”学习”**

- 唯有在概率的意义上, “学习”才是可能的.
- 形象的例子: bin of red and green marbles (用来揭示 probability sense of learning)
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
  - 参考 coin analogy 例子 (书中 Ex 1.10)
- 因此, 对于 final hypothesis $g$, 只能得到:
  - $P[|E_{in}(g) - E_{out}(g)| > \epsilon] \leq 2Me^{-2\epsilon^2N}$
    - M 为 hypothesis 数. 越复杂的模型, M 越大, $E_{out}$ 与 $E_{in}$ 的差距就越大.
    - 常见的模型, 实际上 M 无穷大. 后面会解决这个问题.
  - 从 verification is feasible 到 learning is feasible

**notes**

- 统计学 和 机器学习 的区别之一:
  - 统计学中的理论较严谨, 会做较多或较强的假设
  - 机器学习理论一般更注重实际中的普遍适用性, 减少所需的前提假设


## Homework 1

**作业的意义**
They are meant to make you roll up your sleeves, face uncertainties, and approach the problem from different angles.

Q3: We have 2 opaque bags, each containing 2 balls. One bag has 2 black balls and the other has a black ball and a white ball. You pick a bag at random and then pick one of the balls in that bag at random. When you look at the ball, it is black. You now pick the second ball from that same bag. What is the probability that this ball is also black?

Q7-10: Perceptron Learning Algorithm → [perceptron.ipynb](https://github.com/sunoonlee/machine-learning/blob/master/perceptron/perceptron.ipynb)


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


- $\nabla E_{in}(w) = \frac{2}{N} X^T(Xw - y)$
- 令 $\nabla E_{in}(w) = 0$ , 则 $X^TXw = X^Ty \Rightarrow w = X^{\dagger}y$ 
  - $X^{\dagger} = (X^TX)^{-1}X^T$(pseudo-inverse of X)
  - X is a tall matrix. 一般 $X^TX$ 是可逆的.
- // 这种 Normal Equation 方法的复杂度是 $O(n^3)$ , 当 n 很大时, 效率不如梯度下降法 ($O(kn^2)$)
- // $\frac{\partial(Xw)}{\partial w} = X^T$

**nonlinear transformation**

- algorithms work because of **linearity in weights** (not in inputs/data) 
- transform the data nonlinearly to get new features
  - features are high level representations of raw inputs
- nonlinear transformation
  - $\mathbf{x \rightarrow z}$
  - $g(\mathbf{x}) = sign(\tilde \mathbf{w} \Phi(\mathbf{x}))$.  权重作用于变换后的空间 $\mathcal{Z}$ 而非原始输入空间 $\mathcal{X}$
  - transformation 若选择不当, 会造成泛化的困难


## Lec4 Error and Noise

**Error measures**

- **也叫 error function / cost / objective / risk**
- to quantify how well h approximates f
- $E_{in}(h) = 1/N \sum_{n=1}^N e(h(x_n), f(x_n))$
- $E_{out}(h) = \mathbb{E}_x [e(h(x), f(x)]$ (期望)
- 理想情况下, **error measure should be specified by the user.**
  - 如二分类问题, TN 和 FP 的权重随场景而不同
- 但实际中, 不一定有这种理想的 error measure. 原因一是用户本身无法提供, 二是 weighted cost 可能不容易优化. 替代方法:
  - Plausible measures: 如 squared error $\equiv$ Gaussian noise
    - analytically good
  - Friendly measures: 如 closed-form solution, convex optimization
    - easy to use

**Noisy targets**

- 学习的目标一般情况下并不是 deterministic function, 而是一个概率分布 $P(y|x)$.
- noisy target = deterministic target $f(x) = E(y|x)$ + noise $y-f(x)$
- 加入 error measure 和 noisy targets 后的 learning diagram:
https://d2mxuefqeaa7sj.cloudfront.net/s_FA9189E64E3F201143EF4ED387BEEA610619552CD2E7E2AA295C880EFD385019_1496375358080_image.png

- $P(x)$ vs $P(y|x)$
  - $P(x)$ 是一个未知的概率分布. 我们假定 x 服从某种分布, 使 Hoeffding 不等式成立, 从而使学习成为可能.训练数据应该是独立同分布的. 但我们不需要知道 P(x) 究竟是什么.
  - $P(y|x)$ 是 unknown target distribution, 是学习的目标 (注: 不是 $P(x, y)$ ).

**接下来 12 个 lecture 要下一盘很大的棋**

- 我们需要的结果: $E_{out}(g) \approx 0$ . 可拆分为:
  - $E_{out}(g) \approx E_{in}(g)$
  - $E_{in}(g) \approx 0$
- 因此, **本课程的两个主要问题**
  - **能否使** $E_{out}(g)$ **逼近** $E_{in}(g)$ **?  ← Lec 5~8**
  - **能否使** $E_{in}(g)$ **足够小? ← Lec 9-16**


