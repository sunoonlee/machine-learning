# Ch4 Numerical Computation

- http://www.deeplearningbook.org/contents/numerical.html

## 目录

<!-- toc -->

- [4.1 overflow and underflow](#41-overflow-and-underflow)
- [4.2 poor conditioning](#42-poor-conditioning)
- [4.3 gradient-based optimization](#43-gradient-based-optimization)
- [4.3.1 Beyond the gradient: Jacobian and Hessian matrices](#431-beyond-the-gradient-jacobian-and-hessian-matrices)
- [4.4 constrained optimization](#44-constrained-optimization)

<!-- tocstop -->

## 4.1 overflow and underflow
- underflow: 接近0的数直接变成了0. 这样某些函数会出问题, 如 x/0, 或 log 0.
- overflow: 绝对值很大的数被当做 $\infty$ 或 $-\infty$. 进一步运算时, 无穷大数值通常会被转为 NaN
- 一个例子是 $\mathrm{softmax}(x)_i = \frac{\exp(x_i)}{\sum_j \exp(x_j)}$
  - 假设各 x_i 都等于 c. 当 c 取很大的正值时会 overflow, 取很大的负值时会 underflow.
  - 为了 numerical stability, 利用 softmax 平移不变的性质, 转而求 softmax(z), 其中 $z = x - \max_i x_i$
  - 若要计算 log softmax(x), 当分子 underflow 时会出问题. 因此可以为 log softmax 专门写一个 numerical stable 的实现.
- 很多时候底层库可以代为解决此类问题. 如 Theano 可以自动检测和处理常见的 numerically unstable expressions


## 4.2 poor conditioning
- conditioning: how rapidly a function changes with respect to small changes in its inputs.
  - Functions that change rapidly when their inputs are perturbed slightly can be problematic for scientiﬁc computation because rounding errors in the inputs can result in large changes in the output.
- $f(x) = A^{-1} x$. 矩阵 A 的 condition number: $\max_{i,j}|\frac{\lambda_i}{\lambda_j}|$
  - 当 condition number 很大时, matrix inversion is particularly sensitive to error in the input.


## 4.3 gradient-based optimization
- 梯度为0的点: critical points. 有四种可能: local minimum, local maximum, saddle point, 或平直段.
- 方向导数: 函数在方向u上的斜率 (u是单位向量). 等于 $u^T \nabla_xf(x)$. 梯度的反方向上, 方向导数最小(负值). 因此梯度下降也叫 steepest descent.
- 梯度下降: $x' = x-\epsilon\nabla_xf(x)$. 学习率 $\epsilon$ 的取法:
  - 常数
  - 逐渐减小 vanish
  - 计算不同 $\epsilon$ 取值下的 $f(x-\epsilon\nabla_xf(x)$ , 取使目标函数值最小的 $\epsilon$ ← **line search**
- 有些情况下, 可以无需迭代算法, 直接由 $\nabla_xf(x) = 0$ 求出 x
- 梯度下降仅限于连续空间上的优化. 但这种方法可以推广到离散空间, 如 hill climbing 法.


## 4.3.1 Beyond the gradient: Jacobian and Hessian matrices
- Jacobian 矩阵: 向量对向量的偏导
- Hessian 矩阵: 函数二阶偏导的矩阵. 也是函数梯度的 Jacobian 矩阵.
  - $H_{i,j} = H_{j,i}$. H 是实对称矩阵. 分解可以得到一系列实特征值和正交的特征向量.
- 在单位向量 d 方向上的二阶偏导: $d^THd$
  - 当 d 是 H 的一个特征向量时, 二阶偏导就是对应的特征值
  - 否则, 二阶偏导就是各特征值的加权平均. 与d夹角越小的特征向量, 权重越大.
  - 最大和最小的特征值分别决定了最大和最小的二阶偏导
- 二阶泰勒级数展开
  - $f(x) \approx f(x^{(0)}) + (x-x^{(0)})^Tg + \frac{1}{2}(x-x^{(0)})^TH(x-x^{(0)})$
    - 其中 g 为梯度, H 是 $x^{(0)}$ 处的 Hessian 矩阵
  - 令 $x = x^{(0)} -\epsilon g$, 则
  - $f(x^{(0)}-\epsilon g) \approx f(x^{(0)}) - \epsilon g^Tg + \frac{1}{2}\epsilon^2g^THg$
    - $\epsilon g^Tg$ 为根据函数斜率预测的函数值改变量
    - $\frac{1}{2}\epsilon^2g^THg$ 为根据函数曲率需要进行的修正
    - 当 $g^THg > 0$ 时, 使式子右侧减小量最大的最优步长为 $\epsilon^* = \frac{g^Tg}{g^THg}$
      - 最不利情形下, g 与 H 最大特征值对应的特征向量平行, 此时 $\epsilon^* = \frac{1}{\lambda_{max}}$
  - 小结: 当目标函数能较好地用二次函数近似时, Hessian 矩阵的特征值可决定最优的学习率.
- 对一元函数, 二阶导数决定曲率. 
  - 曲率为0时, 函数的减小量等于 梯度*步长. 曲率为正/负时, 函数的减小量小于/大于 梯度*步长.
  - second derivative test: 在一阶导为0时
    - f’’(x) > 0 → 局部最小
    - f’’(x) < 0 → 局部最大
    - f’’(x) = 0 → 无定论. 可能是鞍点或平直段.
- 多元函数的 second derivitive test
  - 在梯度为0的点 (critical point):
    - Hessian 为正定 → 局部最小
    - Hessian 为负定 → 局部最大
    - Hessian 的特征值至少有一正一负 → 鞍点 (见下图)
    - 当存在0特征值, 且非零特征值全部同号时 → 无定论.

![f(x) = x_1^2 - x_2^2](https://d2mxuefqeaa7sj.cloudfront.net/s_0F75B0043EE5A2DD2B06A6111FFD0DFD6AB8A128E353D878CF131FA4BE2F7368_1503118507118_image.png)

- condition number of the Hessian
  - 多元情况下, 某点上每个方向都有不同的二阶导数值. H 的 condition number 可衡量这些不同的二阶导数值差别的大小.
  - 当 Hessian has a poor condition number 时, 梯度下降法表现很差, 很难选择合适的步长. 见下图.
    - 这里 condition number = 5. 一个方向的曲率是另一方向的5倍, 形成了一个峡谷.
    - 梯度下降法无法利用 Hessian 矩阵包含的信息, 结果浪费了很多时间在峡谷侧壁上移动. 因为最陡的方向不是最优的方向.

![](https://d2mxuefqeaa7sj.cloudfront.net/s_0F75B0043EE5A2DD2B06A6111FFD0DFD6AB8A128E353D878CF131FA4BE2F7368_1503118923543_image.png)

- 牛顿法
  - 可以利用 Hessian 矩阵提高搜索效率. 牛顿法是其中最简单的一种.
  - 对 f(x) 做二阶泰勒展开, 求 critical point, 可得 $x^* = x^{(0)} - H(f)(x^{(0)})^{-1}\nabla_xf(x^{(0)})$
  - 若 f 是正定二次函数, 牛顿法可以一次精确地找到最小值
  - 若 f 不是二次函数, 但局部可以用正定二次函数近似, 那么牛顿法可经过数次迭代找到 critical point (注: 未必是 local minimum), 比梯度下降法快很多
  - 这是一把双刃剑. 当附近的临界点是 local minimum 时就很有利; 而如果附近的临界点是鞍点, 反而有害.
- 一阶优化只用梯度, 二阶优化会用 Hessian 矩阵
- 优化算法的 guarantee
  - 大部分情况下深度学习中的优化算法 come with almost no guarantees
  - 引入 Lipschitz continuity 条件, 可以得到 some guarantee
    - Lipschitz continuity 是一种较弱的条件. $\forall x, \forall y, |f(x)-f(y)|\leq \mathcal{L} ||x-y||_2$, L 为常数
  - convex optimization 可以 provide more guarantees
    - convex function 是一种很强的条件: 函数在各点上的 Hessian 矩阵均为半正定. 这样的函数不存在鞍点, 局部最小值必然是全局最小值.
    - 不过在深度学习中, convex optimization 的应用很有限.


## 4.4 constrained optimization
- 约束优化: 在限定的集合S内寻找使 f(x) 取极值的 x. 集合 S 内的点称为 feasible points.
- 一类简单的方法是对梯度下降做某种修正, 使其能够考虑约束条件.
- 另一大类方法是转化为等价的无约束优化问题.
  - 例如, 寻找使 f(x) 最小的具有单位 L2 norm 的 x, 可转化为优化 $g(\theta) = f([\cos\theta, \sin\theta]^T)$. 不过这样的方法需要创造性.
  - 更一般化的一类方法是 KKT 方法
- KKT 方法
  - 首先把集合 S 用等式和不等式约束条件来描述. $S = \{x|\forall i,g^{(i)}(x)=0 \text{ and }\forall j,h^{(j)}(x)\leq 0 \}$
  - 引入 KKT multipiers $\lambda_i, \alpha_j$. 定义广义拉格朗日函数:
    - $L(x,\lambda,\alpha) = f(x) + \sum_i\lambda_i g^{(i)}(x) + \sum_j\alpha_j h^{(j)}(x)$
  - $\min_x\max_\lambda\max_{\alpha,\alpha\geq 0}L(x,\lambda,\alpha)$ 与 $\min_{x\in S}f(x)$ 等价 (前提是 f(x) 不能为无穷大)
    - 约束优化 → 无约束优化
  - 不等式约束: active / inactive
    - 当 $h^{(i)}(x^*) = 0$ 时, 不等式约束 $h^{(i)}$ is active.
    - inactive 的不等式约束的性质: 当移除这个约束后, 原来的解 $x^*$ 仍然是新问题的一个驻点
    - 对任何不等式约束, $\alpha_i \geq 0$ 或 $h^{(i)}(x^*)=0$ 至少有一个成立. 或者说, $\alpha\odot h(x) = 0$ ( $\odot$ 为 element-wise product)
- 约束优化问题的解满足 KKT 条件 (必要非充分条件)
  - 广义拉格朗日函数梯度为0
  - 所有约束均满足 (包括 $\alpha_i \geq 0$ )
  - $\alpha\odot h(x) = 0$


