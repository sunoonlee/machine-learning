# 机器学习技法 Lec7~11 Aggregation Models

- 课程页面 http://www.csie.ntu.edu.tw/~htlin/mooc/
- 三大部分
  - Lec1~6 embedding numerous features: kernel models. 主要讲 SVM
  - Lec7~11 **combining predictive features: aggregation models ← 暂时只学习这部分**
- lec12~15 distilling hidden features: extraction models. 深度学习, rbf网络, 矩阵分解
  - Lec16 happy learning
# combining predictive features
- blending and bagging
- adaptive boosting
- decision tree
- random forest
- gradient boosted decision tree


## Lec7 blending and bagging

### motivation of aggregation

- 如果有一堆不同的 final hypotheses $g_t$, 怎么决策?
  - selection (by validation): 选一个最好的
  - uniformly vote
  - non-uniformly vote
  - combine the predictions conditionally
    - $G(x) = sign(\sum_{t=1}^T q_t(x) g_t(x))$, with $q_t(x) \geq 0$
- selection vs. aggregation
  - **selection** 是强者主导. **validation** 是一种方式.
  - aggregation 是集体智慧, 有可能弱弱联合变强
- aggregation 如何发挥作用? 两个小例子
  - 左图: 可以用三个简单模型 (水平/竖直分界面) 结合起来, 实现较复杂的分类效果 (中间的两个小矩形都是投票 2:1 取胜) → 类似特征变换?
  - 右图: 对 PLA 的不同结果取 uniform vote, 使结果偏于中庸, 起到类似 large margin 的正则化效果

![](https://d2mxuefqeaa7sj.cloudfront.net/s_813C17C8514357AC0484FF913CDF0EFDCB12EB2863686E09628ED6CC4DAE0254_1502444731405_image.png)


### uniform blending

- 不同的 $g_t$ 越有多样性, blending 效果越明显
- 分类问题: 每人一票
  - 在有争议的点上, 少数服从多数
- 回归问题: 取平均
  - 可以证明 uniform blending 的 MSE 比 $g_t$ 的平均表现要好
    - 对任一 x: $avg((g_t - f)^2) = avg((g_t - G)^2) + (G-f)^2$
    - 对 x 取期望: $avg(E_{out}(g_t)) = avg(\varepsilon(g_t - G)^2) + E_{out}(G) \geq E_{out}(G)$
  - 这个分解在下面这种**虚拟过程**下, 就是 LFD 里讲过的 bias-variance 分解:
    - 对数据背后的分布进行 T 次不同的采样, 分别得到数据集 $D_t$
    - 对 $D_t$ 实施同一算法, 得到 $g_t$. 
    - 取平均得到 G. 取极限: $\bar{g} = \lim_{T\rightarrow \infty}G$.
    - $avg(E_{out}(g_t)) = avg(\varepsilon(g_t - \bar{g})^2) + E_{out}(\bar{g})$
      - 第2项是 bias: performance of consensus (共识)
      - 第1项是 variance: expected deviation to consensus

### Linear and any blending

- linear blending:
  - 对不同的 $g_t$ 做线性组合
  - linear blending = Linear model + hypotheses as transform
  - 用于回归时, 跟采用特征变换的线性回归很像

![](https://d2mxuefqeaa7sj.cloudfront.net/s_813C17C8514357AC0484FF913CDF0EFDCB12EB2863686E09628ED6CC4DAE0254_1502448960528_image.png)

  - two-level learning
    - 第一层是独立地学得不同的 $g_t$ 
    - 第二层是以 $g_t(x)$ 为 “特征”, 用线性模型进行”学习” ← linear blending
  - 与 validation 类似, linear blending 应该在验证集上学习. 否则会在 d_vc 上付出很大代价
- any blending (stacking): 更一般化的 blending 方法
  - 流程:
    - 从训练集分别学得 $g_1^-, g_2^-, ..., g_T^-$ 
      - 上标表示未使用验证集数据. 与 validation 类似, 在确定 blending 函数后, 可以把验证集数据放回来重新训练得到 $g_1, g_2, ...,g_T$.
    - 对验证集数据做变换: 由 $(x_n,y_n)$ 变换为 $(z_n=\phi^-(x_n),y_n)$, 其中 $\phi^-(x) = (g_1^-(x),...,g_T^-(x))$
    - 使用变换后的验证集数据 $\{(z_n,y_n)\}$确定 blending “模型” $\tilde{g}$
      - $\tilde{g}$ 选为线性函数时, 就是 linear blending
    - 最终模型 $G(x) = \tilde{g}(\phi(x))$  ← 两层模型的嵌套
      - 注意这里的 $\phi$ 不再有上标 -
  - any blending 比较强大, 也容易过拟合, 需小心使用

### bagging (bootstrap aggregation)

- **注: bagging 名称的由来是 Bootstrap AGGregatING**
- 对于 uniform aggregation, diversity 很重要. **diversity** 可以是源于:
  - 不同的 models (hypotheses set H)
  - 不同的超参数
  - algorithmic randomness, 如随机初始化
  - data randomness, 如 cross-validation.
  - 这一节涉及另一种利用 data randomness 获得 diversity 的方法
- 前面讲到一种虚拟的 aggregation, 是对数据背后的分布进行假想的多次采样, 得到不同的 $D_t$. 现实是我们只有一个固定的 D. 而 bootstrap aggregation 就是要从这个固定的 D 获得多个不同的数据集 $D_t$
- **boostrapping**: a statistical tool that re-samples from D to “simulate” $D_t$
  - 具体来说, 是用均匀的**放回**采样. 采样的数量等于或不等于 N 都可以.
- bagging: 一种简单的 **meta algorithm**
  - 先从不同 $D_t$ 用 **base algorithm** A 学得不同的 $g_t$
  - 对这些 $g_t$ 取 uniform aggregation 得到 G
- 当 base algorithm 对数据随机性敏感时, bagging 表现较好

### summary

- blending vs. bagging
  - blending 是一种 after-the-fact aggregation.
  - bagging 的特点是对数据集的 re-sample (样本扰动).
- bagging vs. boosting
  - bagging 是并行式的, boosting 是串行式的
  - 二者都是对样本的某种 re-weighting
## Lec8 Adaptive boosting

### motivation of boosting

- 一个浅显的例子: 老师带领学生识别苹果
- 有点像自动的 error analysis ?

### diversity by re-weighting

- boostrapping 实际上也是一种 re-weighting
  - 抽到多次的样本, 权重 > 1; 没抽到的, 权重 = 0.
- weighted base algorithm
  - $E_{in}^u(h) = \frac{1}{N} u_n err(y_n,h(x_n))$  ← weighted error
  - LFD 中讲过 class-weighted learning, 根据实际场景, 给不同的 class 不同权重. 现在讲的是 example-weighted learning, 可以看做是一种延伸.
- 怎样的 re-weighting 策略可以得到 more diverse hypotheses?
  - 具体来说, 若第 t 个 hypothesis $g_t$ 已定, 选什么样的权重 $u^{(t+1)}$可以使 $g_{t+1}$ 与 $g_t$ 尽可能地 diverse?
  - 思路: 使以 $u^{(t+1)}$为权重时 $g_t$ 的表现接近 random, 即: $\frac{\sum_n u_n^{(t+1)}I(y_n\neq g_t(x_n))} {\sum_n u_n^{(t+1)}} = 1/2$  ← “optimal” re-weighting
  - 实现: 由 $u^{(t)}$ re-scaling 得到 $u^{(t+1)}$. re-scaling 的系数与 $g_t$ 的错误率 $\epsilon_t$ 有关

### Adaptive boosting algorithm

- AdaBoost 包括三个要素:
  - weak base algorithm (Student)
  - optimal re-weighting factor (Teacher)
  - linear aggregation $\alpha_t$ (Class)

![](https://d2mxuefqeaa7sj.cloudfront.net/s_813C17C8514357AC0484FF913CDF0EFDCB12EB2863686E09628ED6CC4DAE0254_1502520340997_image.png)

- AdaBoost 步骤
  - 初始权重: 均匀地取 1/N
  - 循环
    - 基于权重 $u^{(t)}$ 学得 $g_t$
    - re-weighting: $u^{(t)}$ → $u^{(t+1)}$
    - 计算 linear aggregation 系数 $\alpha_t$ ← linear aggregation “on the fly”
  - 线性组合 $g_t$ 得到 G
- re-weighting 是关键, 使每一步的学习更多地关注上一步的 g 的错误.
- AdaBoost 的优良性能 (理论上有保证)
  - base 算法可以很弱, 只要比 random 略好就行 ( $\epsilon_t < 1/2$ )
  - 经过 T = O(log N) 轮循环, 就可以使 $E_{in}(G) = 0$; 同时泛化性能也比较好
  - 使一群臭皮匠达到诸葛亮的水平. 这个算法中文有译作 “皮匠法” :D

### AdaBoost in action

- 一个例子: AdaBoost-Stump
  - 一种很弱的 base algorithm: desicion stump. 只考虑一个特征, 水平/竖直切一刀
  - 利用 AdaBoost, 可以实现高效的 feature selection 和 aggregation
    - 因为 desicion stump 每次只取一个特征, 所以循环 T 次后, 就自动选出了 < T 个特征 (有些特征因为没效果, 算出 $\alpha_t = 0$, 就被自动抛弃了)

![](https://d2mxuefqeaa7sj.cloudfront.net/s_813C17C8514357AC0484FF913CDF0EFDCB12EB2863686E09628ED6CC4DAE0254_1502521937494_image.png)

- 前后章节关系
  - bagging 是 uniform aggregation
  - AdaBoost 是 linear aggregation
  - 下一讲的 desicion tree 是 conditional aggregation

### 补充一点集成学习基本知识 (西瓜书 ch8)

- 集成学习: ensemble learning, 也即 aggregation. 是将 individual learner 用某种策略结合起来
  - 集成可以是同质的, 也可以是异质的. 同质集成的 individual learner 也叫 base learner
- 大致分为两类
  - 个体学习器之间存在强依赖关系, 必须串行生成的序列化方法. 代表是 boosting
  - 个体学习器之间不存在强依赖关系, 可同时生成的并行化方法. 代表是 bagging 和随机森林.
    - 在 bagging 基础上, 随机森林通过小的改进, 增加了 base learner 的多样性, 从而提升了泛化性能
- 从 bias-variance 分解的角度, boosting 主要关注降低 bias, bagging 主要关注降低 variance


## Lec9 Decision tree

### decision tree hypothesis


![](https://d2mxuefqeaa7sj.cloudfront.net/s_813C17C8514357AC0484FF913CDF0EFDCB12EB2863686E09628ED6CC4DAE0254_1502683715537_image.png)

- path view: $G(x) = \sum_t I(\text{x on path t})\cdot leaf_t(x)$
- recursive view: $G(x) = \sum_c I(b(x) = c) \cdot G_c(x)$
- 李航书: 可以认为是 if-then 规则的集合, 也可以认为是定义在特征空间和类空间上的条件概率分布
- 优点: 人类可读; 简单; 训练和预测高效
- 缺点: 理论根据少, heuristics 多. 没有唯一的 representative algorithm.

### decision tree algorithm

- 决策树算法的一般递归实现

![](https://d2mxuefqeaa7sj.cloudfront.net/s_813C17C8514357AC0484FF913CDF0EFDCB12EB2863686E09628ED6CC4DAE0254_1502704073977_image.png)

- 决策树算法的四项设计: 
  - 分支数 C
  - base hypothesis, 即叶节点返回的 $g_t$
  - 分支准则
  - 终止准则
- 一种常见的决策树算法 C&RT (classification and regression tree, 也叫 CART)
  - C = 2
  - base hypothesis: 使 E_in 最优的常数. 
    - 分类问题返回占多数的标签 y, 回归问题返回y的均值.
  - **分支准则: purifying**
    - 采用使两边总纯度最低的 desicion stump (总纯度: 两边的纯度按样本数加权求和)
    - 纯度的衡量 inpurity function: 分类常选基尼指数, 回归选均方误差
  - 终止准则: 有两种
    - 节点 y_n 相同: 没法再纯了
    - 节点 x_n 相同: 没法再分了
  - 简言之, CART 就是 **fully-grown tree** with **constant leaves** that come from **bi-branching** by **purifying**

### desicion tree heuristics in C&RT

- CART 算法处理多分类问题很方便
- **regularization by pruning**
  - fully-grown tree 可以做到 E_in = 0, 但往往会过拟合. 因为靠近叶子的节点上数据量已经比较小了.
  - 可引入正则化项 $\Omega(G) = \mathrm{NumOfLeaves}(G)$.
  - regularized decision tree: $\mathrm{argmin}_{\text{all possible G}} E_{in}(G) + \lambda \Omega(G)$
  - 但穷举所有的 G 几乎不可能. 简化的做法是: 从 fully-grown tree $G^{(0)}$ 开始, 依次减去一个叶子, 得到 $G^{(1)}, G^{(2)}, ...$
- 处理 categorical features
  - numerical feature 用的是 decision stump: $b(x) = I(x_i \leq \theta) + 1$
  - categorical feature 可以用 **decision subset**: $b(x) = I(x_i \in S) + 1$, 其中 S 是 {1,2,…K} 的子集
- 对 missing features 使用 surrogate branch

### decision tree in action

- CART 与 AdaBoost-Stump 的一个小对比
  - AdaBoost-stump 的每一刀都是切在整个特征空间上的, 而 CART 是切在某个条件下的子空间内的, 未必会穿越整个特征空间
  - Adaboost_stump 在较早期的步骤里决策界面就有了大致形状, 后面的步骤好似在微调; CART 因为是递归的, 到后期才会成形.

![](https://d2mxuefqeaa7sj.cloudfront.net/s_813C17C8514357AC0484FF913CDF0EFDCB12EB2863686E09628ED6CC4DAE0254_1502707438275_image.png)

- CART 的实用优点
  - 人类可读; 易处理多分类问题; 易处理类别型特征; 易处理缺失特征; 高效的非线性训练
  - 几乎没有其他算法同时做到以上这些 (除了其他的决策树算法 如 C4.5)


## Lec10 Random forest

### random forest algorithm

- 基本的随机森林算法
  - **RF = bagging + fully-grown CART decision tree**
  - 决策树 variance 较大 (尤其是 fully grown 的). bagging 可降低 variance
  - “随机” 表示为了增加 base hypothesis 的 diversity 而引入的各种随机性; “森林” 表示多棵决策树.
- 进一步, 可以对特征抽样. 于是 **RF = bagging + random-subspace CART.**
  - 训练**每棵树**时, 从全部 d 个特征中, 随机抽出 d’ 个, 相当于特征空间由 d维空间 变成一个 d’维随机子空间.
    - 常选 d’ << d. 这样计算效率较高.
    - 这种思想也可以应用到其他算法中
  - original RF 算法在每棵树**每次分叉**时都要 re-sample new subspace.
- 再进一步, 对特征进行随机组合, 或者说 projection. 于是 **RF = bagging + random-combination CART**
  - 训练每棵树时, 采用的特征是 $\Phi(x) = P\cdot x$. 其中矩阵 P 的每个行向量 $p_i$ 定义了一个新特征 $\phi_i(x) = p_i^Tx$
    - 前面的 random subspace 就是 $p_i$ 为 natural basis 的特例
    - 常选 d’ << d, 即投影到低维空间
  - original RF 算法在每次分叉时都要做一次这样的 random-conbination

### out-of-bag estimate

- 在 bootstrapping 中, 每个 base hypothesis $g_t$ 都有一部分样本没有用到. 这部分样本叫做 $g_t$ 的 out-of-bag (OOB) examples
- 假设 bootstrapping 抽样的数量 N’ = N, 那么一个样本是 $g_t$ 的 OOB 的概率是 $(1-\frac{1}{N})^N$, 当 N 很大时约为 1/e
- 与 validation 类似, OOB 可用作 self-validation of bagging/RF, 用来确定超参数等
  - $E_{oob}(G) = \frac{1}{N} \sum_{n=1}^N err(y_n, G_n^- (x_n))$
    - 其中, $G_n^-$ 表示没有抽到第 n 个样本 (即 $(x_n, y_n)$ ) 的 base hypothesis 的平均
- OOB 与 validation 的区别
  - OOB examples 自然产生于 bagging 过程, 不需要像 validation 那样专门拿出一部分数据
  - validation 检验 $g_i^-$ (并从中选择最优), OOB 检验 G
  - OOB 不像 validation 那样需要 re-training

### feature selection

- 特征选择有可能提高模型的效率, 泛化能力和可解释性, 但也有可能因选择不当而造成过拟合和错误解释.
- 可以为每个特征计算出 importance 指标, 根据该指标来选择特征
  - 线性模型特征的 importance 很容易计算, 可以取对应参数的绝对值. 非线性模型需要另想办法.
- feature importance by permutation test
  - 衡量重要性的思路: 把某一个特征 $x_i$ 的取值全部改为”随机值”, 看算法表现退化多少
  - 怎么取”随机值”? 如果直接取为服从高斯分布或其他分布的随机值, 实际上改变了 $x_i$ 的分布. 更好的选择是 permutation.
  - permutation test 是一种常见的统计学工具, 可用于类似 RF 的任何非线性模型
  - importance(i) = performance(D) - performance(D^(p)). $D^{(p)}$ 表示把 D 中所有样本的特征i取值进行随机重排.
- 原版随机森林算法中的 feature importance 定义
  - 前面提到的 $\mathrm{performace}(D^{(p)})$ 需要 re-training, 比较麻烦
  - original RF 采用一种简化的定义: $\mathrm{importance}(i) = E_{oob}(G) - E_{oob}^{(p)}(G)$
    - 区别是不用重新去学习 $D^{(p)}$ 下的 G, 直接用已有的 G.
- 总之, RF 可以通过 permutation + OOB 实现 feature selection, 非常高效和有用

### random forest in action

- 主要讲了三个例子. 最后讲了一下如何确定树的数量.

![](https://d2mxuefqeaa7sj.cloudfront.net/s_813C17C8514357AC0484FF913CDF0EFDCB12EB2863686E09628ED6CC4DAE0254_1502795222200_image.png)


![](https://d2mxuefqeaa7sj.cloudfront.net/s_813C17C8514357AC0484FF913CDF0EFDCB12EB2863686E09628ED6CC4DAE0254_1502795255852_image.png)


![](https://d2mxuefqeaa7sj.cloudfront.net/s_813C17C8514357AC0484FF913CDF0EFDCB12EB2863686E09628ED6CC4DAE0254_1502795240409_image.png)



- 三个例子
  - 从例1可以看出, RF 得到的边界较为平滑, 而且有种 Large margin 的效果
  - 从例3可以看出, 通过多棵树的 voting, 可以消除噪音点的影响
- 树的数量
  - 越多越好. 2013 KDDCup 冠军队用了 12000 trees
  - 当整个随机过程不稳定时, 树的数量更为重要. 这算是 RF 的一个缺点. 实际中要特别注意检查 G 的稳定性, 保证有足够多的树.


## Lec11 GBDT

### adaptive boosted decision tree

- AdaBoost + sampling $\propto u^{(t)}$ + DTree( $\tilde{D}_t$ )
  - 把权重与 base 算法解耦, 通过对数据集 sampling 来实现
- 如果用全部数据训练 full grown tree, 会得到 $\epsilon_t = 0$, 导致 $g_t$ 的组合系数 $\alpha_t$ 为无穷大. 可从两方面解决:
  - 剪枝: 一般的剪枝方法, 或者限制树高
  - 不使用全部数据: sampling 就可以做到这一点
- 当 AdaBoost-Dtree 限制树高 = 1, 且不做 sampling 时, 就得到了特例 Adaboost-stump

### optimization view of AdaBoost

- 这一节数学内容较多, 费了许多周折, 就是想说明 AdaBoost 跟某种梯度下降的联系. 
- example weights of AdaBoost
  - 前面讲过的 AdaBoost weights 更新过程可以表达为更数学化的形式:
    -  $u_n^{(T+1)} = \frac{1}{N} \exp(-y_n\sum_{t=1}^T\alpha_t g_t(x_n))$
    - // 二分类问题, 当 label 为 {-1, 1} 时: $y_nh(x_n)$ = 1 (分类正确), 或-1 (分类错误)
    - $\sum_{t=1}^T\alpha_t g_t(x)$ : voting score, 记为 s
- voting score and margin
  - 类比 SVM, $y\cdot s$ 即为 signed & unnormalized margin
  - 因此, 减小 $u_n^{(T+1)}$ 相当于增大 margin
- AdaBoost error function
  - 记 $s_n = \sum_{t=1}^T\alpha_t g_t(x_n)$
  - AdaBoost 可看作是在优化这个目标: $\sum_{n=1}^N u_n^{(T+1)} = \frac{1}{N} \sum_{n=1}^N \exp(-y_ns_n)$
  - 其中, $\exp(-ys)$ 是 exponential error measure, 是 0-1 error 的一个上界
- Gradient descent on AdaBoost Error Function
  - 过程没太看懂. 大致是说, AdaBoost 相当于一个梯度下降过程, $\min_\eta\min_h \hat{E}_{ADA} = \sum_n u_n^{(t)} \exp(-y_n\eta h(x_n))$
    - 求解 $g_t$ 相当于寻找最优方向 h
    - 求解组合系数 $\alpha_t$ 相当于寻找最优步长 $\eta$ (steepest descent)
  - AdaBoost 即为: steepest descent with approximate functional gradient

### Gradient boosting

- 上一节的结论是 AdaBoost 可看做某种特殊 error function 的梯度下降.
- 推广到一般的 error function, 就是 GradientBoost, 可以用于回归或 soft 分类问题.

![](https://d2mxuefqeaa7sj.cloudfront.net/s_813C17C8514357AC0484FF913CDF0EFDCB12EB2863686E09628ED6CC4DAE0254_1502866115157_image.png)

- 例如: 对回归问题, error function 可取平方误差. 整个算法流程为:

![](https://d2mxuefqeaa7sj.cloudfront.net/s_813C17C8514357AC0484FF913CDF0EFDCB12EB2863686E09628ED6CC4DAE0254_1502866421042_image.png)

- GBDT 可看做 AdaBoost-DTree 的回归版本

### summary of aggregation models

- aggregation models 大体上可分为两大类
  - blending models: aggregate **after** getting diverse $g_t$.
  - **aggregation-learning** models: aggregate **as well as** getting diverse $g_t$
  - 前者比较简单, 一般很少提及. 所以一般情况下 aggregatio 特指后者.
- 四种 aggregation models

| model          | diverse $g_t$ by | voting 方式                      | 以 DTree 为 base learner       |
| -------------- | ------------------ | ------------------------------ | ---------------------------- |
| bagging        | bootstrapping      | uniform vote                   | strong DTree → random forest |
| AdaBoost       | reweighting        | linear vote by steepest search | weak DTree → AdaBoost-DTree  |
| Gradient Boost | residual fitting   | linear vote by steepest search | weak DTree → GBDT            |
| decision tree  | data splitting     | conditional vote by branching  | -                            |

- 两类 aggregation 方法的功效完全不同
  - 一类是降低 bias, 解决较弱的 base learner 的欠拟合问题, 使弱弱联合变强, 效果类似 feature transform. 例子是 Boost.
  - 一类是降低 variance, 解决较强的 base learner 的 overfitting, 使其组合后趋于中庸. 效果类似 regularization. 例子是 bagging 和  random forest.







