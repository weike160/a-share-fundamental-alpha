# A 股基本面事件 Alpha 研究

**A-share Fundamental Event Alpha: Point-in-Time Signals, Trading Frictions and Machine Learning**

这个项目记录一次完整的 A 股量化研究。研究从财报公告后的价格漂移（PEAD）出发，逐步处理数据时点、样本偏差、交易限制、成本和风险暴露，最后再检验机器学习能否带来额外收益。

项目目前处于研究问题阶段，阶段记录见 [`research/01_research_question.md`](research/01_research_question.md)。仓库中的模块已经搭好，但大多仍是会抛出 `NotImplementedError` 的接口骨架，暂时没有可采信的回测结果。

## 当前里程碑：AKShare 公开数据基线

第一条可运行研究链路将使用 AKShare 获取财务指标、实际披露日期、日频行情和停复牌信息。范围暂时收窄为一个 SUE proxy、一组增长指标、5 至 40 个交易日的因子检验，以及带基础交易成本的 long-only 组合。

这个版本的目标是得到一份能从原始输入重新生成的研究结果。AKShare 返回的历史财务报表可能包含后续修订，历史股票池、ST 状态和精确公告时刻也未必能够完整恢复。因此，公开数据基线会单独报告这些限制，不会将结果表述为严格的机构级 Point-in-Time 回测。

实现范围、时间规则和验收条件见 [`research/02_akshare_mvp.md`](research/02_akshare_mvp.md)。

## 完成后可以检查什么

项目最终会交付一套可复现的研究结果，包括数据处理、因子检验、组合回测、成本分析和研究报告。

| 交付物 | 包含内容 | 检查方式 |
|---|---|---|
| Point-in-Time 数据集构建代码 | 历史股票池、公告时间、退市记录、复权信息和可交易状态 | 抽查任意日期，确认模型没有使用当时尚未公开的信息 |
| 因子研究结果 | SUE 定义、覆盖率、RankIC、ICIR、分层收益、衰减曲线和横截面回归 | 更换预测期、股票池或中性化设置后重新运行 |
| 可交易组合回测 | 持仓、换手、行业与市值暴露，以及每期收益 | 从信号重新生成权重和净值，并核对约束是否生效 |
| 成本与容量分析 | 佣金、印花税、滑点、市场冲击，以及 Gross Alpha 到 Net Alpha 的拆解 | 调整成交参与率和成本参数，观察结果如何变化 |
| 稳健性与失败记录 | 子时期、市场环境、持有期、参数敏感性和失效场景 | 查看包含全部配置的实验矩阵 |
| 机器学习增量实验 | 线性模型、LightGBM 和 MLP 的样本外对比与特征解释 | 使用相同时间切分和成本口径比较各模型 |
| 研究报告 | 数据口径、方法、结果、局限和可复现命令 | 从原始输入重新生成主要表格和图表 |

最终结论可能是信号在交易成本后仍然有效，也可能是收益来自风格暴露、微盘股集中，或根本无法覆盖成本。报告会保留这些结果，并说明证据支持哪一种解释。

## 研究问题

A 股公司公布财报后，价格是否会持续反映财报中的新信息？如果严格使用当时可获得的数据，并把停牌、涨跌停、T+1 和交易成本算进去，这种价格漂移是否仍有交易价值？

第一阶段研究 PEAD，并用标准化未预期盈利（SUE）表示财报超预期程度：

```
SUE_{i,t} = (ActualEPS - ExpectedEPS) / σ(EarningsSurprise)
```

待检验的关系是：

```
SUE ↑  ⇒  FutureReturn ↑
```

也就是比较财报超预期程度与公告后收益。如果暂时拿不到可靠的分析师预期数据，会先测试这些代理变量：

- 净利润同比变化
- 营收同比变化
- EPS 同比变化
- ROE 和毛利率变化
- 业绩预告超预期 proxy

代理变量只能用于早期探索，不能和基于分析师一致预期计算的 SUE 混为一谈。

## 研究流程

```
金融问题 → 研究假设 → Point-in-Time 数据 → Alpha Signal → 统计检验
     → 组合构建 → 交易约束与成本 → 风险分析与归因
     → 传统因子基线 → 机器学习 → 增量 Alpha 检验
```

研究重点是弄清信号在什么条件下有效、收益来自哪里，以及实际交易会消耗多少收益。单独一个漂亮的 Sharpe 无法回答这些问题。

## 项目结构

```
a-share-fundamental-alpha/
│
├── data/                       # Point-in-Time 数据环境
│   ├── universe.py             #   历史时点股票池
│   ├── financials.py           #   财报数据
│   ├── announcement.py         #   公告时间对齐
│   ├── corporate_action.py     #   分红、送股、配股、拆分与复权
│   ├── delisting.py            #   退市处理
│   └── tradability.py          #   停牌、涨跌停、T+1 与 ADV
│
├── signals/                    # 基本面事件信号
│   ├── sue.py                  #   Standardized Unexpected Earnings
│   ├── earnings_growth.py      #   净利润和营收同比变化
│   ├── revenue_surprise.py     #   营收超预期
│   └── analyst_revision.py     #   分析师预测修正
│
├── factor/                     # 因子检验
│   ├── preprocess.py           #   清洗、填充、去极值与标准化
│   ├── neutralize.py           #   行业和市值中性化
│   ├── ic.py                   #   IC、RankIC 与 ICIR
│   ├── decay.py                #   不同预测期的 IC Decay
│   └── quantile.py             #   Q1 至 Q10 分层收益
│
├── portfolio/                  # 组合构建与风险指标
│   ├── construction.py         #   Long-only benchmark-relative 组合
│   ├── optimizer.py            #   权重、暴露与换手约束
│   └── risk.py                 #   Sharpe、MaxDD、TrackingError 与 IR
│
├── execution/                  # 交易成本模型
│   ├── transaction_cost.py     #   佣金和印花税
│   ├── slippage.py             #   滑点
│   └── market_impact.py        #   基于 OrderSize/ADV 的冲击成本
│
├── backtest/
│   └── engine.py               #   事件与组合回测引擎
│
├── ml/                         # 机器学习实验
│   ├── linear.py               #   线性模型基线
│   ├── lightgbm.py             #   LightGBM / XGBoost
│   └── neural_net.py           #   MLP
│
├── research/
│   ├── 01_research_question.md #   研究问题与证伪条件
│   └── 02_akshare_mvp.md       #   AKShare 最小闭环计划
│
├── tests/                      # 当前只有包导入冒烟测试
├── report/                     # 研究报告目录
├── README.md
└── pyproject.toml
```

## 分阶段计划

### 1. 定义问题

先写清 PEAD 的经济动机、核心假设和可检验指标。可能的解释包括信息反应不足、关注度差异和套利限制，但这些解释需要由数据区分。

### 2. 建立 Point-in-Time 数据集

- 按历史时点还原股票池，保留后来退市、被收购或进入 ST 状态的公司。
- 使用真实公告时间，并确保 `InformationTime < TradingTime`。
- 处理上市、退市、停牌、涨跌停、复权、T+1 和印花税。
- 用 ADV 约束成交规模并估算策略容量。

### 3. 构造信号

实现 SUE 和增长类基本面信号。每个信号都要有明确的数据来源、计算公式、可用时间和缺失值处理规则。

### 4. 检验因子

- 计算 IC、RankIC、ICIR 和 Q1 至 Q10 分层收益，检查分层是否单调。
- 观察 1 至 60 个交易日的 IC Decay。
- 做行业、市值中性化和横截面回归，检验 `β₁(SUE)`。
- 按市值、流动性、波动率和分析师覆盖度拆分样本。

### 5. 构建组合

建立 long-only benchmark-relative 组合，并约束单只股票权重、行业暴露、市值暴露、换手率和流动性。

### 6. 加入交易限制和成本

模拟佣金、印花税、滑点和市场冲击，拆解 Gross Alpha 与 Net Alpha 的差异。

### 7. 分析风险和收益来源

报告 AnnualReturn、Sharpe、MaxDD、Turnover、TrackingError 和 InformationRatio，并检查收益是否集中于少数年份、行业或小市值股票。

### 8. 做稳健性与失败检验

更换股票池、子时期、市场环境、持有期、成本假设和信号定义。如果参数稍作调整，结果就消失，应把它视为过拟合风险，而不是继续寻找更好看的参数。

### 9. 检验机器学习的增量

计划使用以下特征：

```
[SUE, RevenueGrowth, ProfitGrowth, Momentum, Volatility,
 Liquidity, Size, Industry, MarketRegime]
```

模型从线性基线开始，再测试 LightGBM 和 MLP。评价指标包括 RankIC、ICIR、Net Sharpe、Turnover、MaxDD 和 Capacity。只有样本外表现稳定，而且提升并非来自更高换手或过度集中于微盘股，才能算作增量 Alpha。模型解释暂定使用 SHAP 和 PDP。

## 验收标准

### Level 1：数据正确

- [ ] 使用历史股票池并保留退市股票
- [ ] 正确处理上市与退市日期
- [ ] 使用真实公告时间，避免 look-ahead bias
- [ ] 正确处理复权、停牌和涨跌停
- [ ] 明确 ST / *ST 股票的处理规则

Level 1 未完成前，不评价任何回测收益。

### Level 2：研究设计成立

- [ ] 完成 Signal 定义、RankIC、ICIR、分层收益和 IC Decay
- [ ] 完成行业与市值中性化、横截面回归、分时期检验和机制检验

### Level 3：结果可以落到交易

- [ ] 完成组合构建、换手率、佣金、印花税和滑点模拟
- [ ] 加入流动性限制与市场冲击
- [ ] 报告 Gross vs Net、MaxDD 和 Capacity

### Level 4：机器学习有样本外增量

- [ ] 样本外 RankIC、ICIR 或 Net Sharpe 相对基线有所改善
- [ ] 改善出现在多个年份，且并非由更高换手或微盘股集中造成
- [ ] 扣除交易成本后仍有增量

## 结果如何呈现

主报告至少给出以下内容：

- 数据覆盖范围、缺失情况和样本筛选过程
- SUE 的分布、覆盖率、RankIC、ICIR、分层收益和 IC Decay
- 中性化与横截面回归结果，包括系数、显著性和稳定性
- Gross 与 Net 收益、换手率、回撤、跟踪误差和容量估计
- 按年份、股票池、市场环境和关键参数拆分的稳健性结果
- 失败实验、异常结果及其排查过程
- 线性基线与机器学习模型在同一时间切分下的样本外比较

结果摘要会把假设归入“得到支持”“证据不足”或“被当前实验否定”，并附上对应的统计结果和交易假设。项目是否完成取决于研究链条能否复现，`Sharpe > 2` 不在完成条件之内。
