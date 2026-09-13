# A 股基本面事件 Alpha 研究

**A-share Fundamental Event Alpha: Point-in-Time Signals, Trading Frictions and Machine Learning**

这个项目记录一次完整的 A 股量化研究。研究从财报公告后的价格漂移（PEAD）出发，逐步处理数据时点、样本偏差、交易限制、成本和风险暴露，最后再检验机器学习能否带来额外收益。

> 版本：`v1.0`，最小可行研究闭环，已跑通全链路。
>
> `v1.0` 使用 AKShare 公开数据完成第一条可复现链路，从财报事件面板开始，经过信号构造和统计检验，最终得到加入基础交易成本的 long-only 组合结果。复杂组合优化和精细市场冲击校准留给后续版本。

## 当前完成情况

- 已定义 PEAD 研究问题、核心假设、评价指标和证伪条件。
- 已完成数据、信号、因子、组合、执行、回测和机器学习模块的实现，配套离线测试 `201 passed`。
- 已下载并缓存 22 个报告期（2019Q1–2024Q2）的财务、披露日期与日频行情，得到 **91,598 条财报事件、4,953 只股票**。
- 已完成 SUE 信号（覆盖率 96.0%）、因子检验、扣成本回测、稳健性检验与机器学习样本外比较。
- **主假设在本样本上被否定**：SUE 的 10/20 日 RankIC 显著为负，扣成本后组合年化 -6.1%，跑输等权事件基准。结论与限制见 [`report/REPORT.md`](report/REPORT.md)。

研究问题记录在 [`research/01_research_question.md`](research/01_research_question.md)，`v1.0` 的实现计划见 [`research/02_akshare_mvp.md`](research/02_akshare_mvp.md)，研究配置见 [`research/config.py`](research/config.py)。

## v1.0 产出

同一条命令 `python scripts/run_all.py` 生成以下产出：

| 交付物 | 文件 | 检查方式 |
|---|---|---|
| 数据快照与审计表 | `results/data_audit.csv`、`sample_audit.md`、`return_audit.csv`、`analysis_audit.csv` | 检查披露日期、信号可用日期和首次交易日期 |
| 财报事件面板 | `data/processed/` 下的 `event_panel`、`event_panel_returns`、`event_panel_signal`、`analysis_panel` | 从缓存数据重新生成并核对样本筛选过程 |
| 因子研究结果 | `results/factor_metrics.json`、`ic_by_period.csv`、`quantile_returns.csv`、`cross_sectional_regression.csv` | 更换预测期或中性化设置后重新运行 |
| long-only 回测 | `results/backtest_cohorts.csv`、`backtest_metrics.json` | 从信号重新生成权重与净值 |
| 成本与稳健性结果 | `results/robustness.csv`、`results/figures/gross_to_net.png` | 比较 Gross 与 Net 结果，并保留失效配置 |
| 机器学习比较 | `results/model_comparison.csv`、`ml_robustness.csv` | 同一时间切分下与线性基线比较 |
| 可复现报告 | `report/REPORT.md`、`results/figures/*.png` | 在新环境中重新生成主要结果 |

AKShare 返回的历史财务报表可能包含后续修订（试点扫描 60 只股票，发现 4 家财报被更正），历史股票池、ST 状态和公告精确时刻也未必能够完整恢复。`v1.0` 会把这些问题写入数据审计和结果限制，不将公开数据基线表述为严格的机构级 Point-in-Time 回测。

## 主要结果（v1.0，2019Q1–2024Q2）

| 环节 | 结果 |
|---|---|
| 样本 | 91,598 事件 / 4,953 只股票 / 22 个报告期，时序违规 0 条 |
| SUE 覆盖率 | 96.0%（同花顺 EPS 与东财业绩报表口径一致率 99.93%） |
| RankIC（20 日） | **-0.036**，t = -2.54；5 日 -0.025、10 日 -0.035（均为负） |
| 分层收益（20 日） | 单调**递减**：Q1 +2.88% → Q5 +1.74% |
| 横截面回归 | 控制市值/动量/波动后 SUE 系数不显著（平均 t = -0.66） |
| long-only 回测 | Gross 年化 +2.0%，**Net 年化 -6.1%**，净 Sharpe -0.21 |
| 等权事件基准 | 年化 +12.9%，Sharpe 0.65；策略 IR **-1.50** |
| 交易成本 | 68.3 bp/期（佣金 6.0 + 印花税 5.0 + 滑点 29.2 + 冲击 28.2） |
| 容量 | 中位数约 4.9 亿元（10% ADV 参与率） |
| 稳健性 | 35 个情景中扣成本后年化为正的比例 22.9%（8 个），**没有任何情景跑赢等权事件基准**；分年份正负翻转 |
| 机器学习（样本外） | LightGBM RankIC 0.109、净年化 +23.5%；**去掉 SUE 特征后几乎不变** |

研究结论由实际结果决定。本样本的结论是：以季节性随机游走 SUE 代理衡量的 PEAD 在扣成本后**不构成可交易 Alpha**；高 SUE 组合反而跑输等权事件基准。报告保留了全部失效配置、异常与数据限制。

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
│   ├── source.py               #   AKShare 抓取与原始响应缓存 (可追溯、可重放)
│   ├── financials.py           #   财报数据与字段口径
│   ├── announcement.py         #   真实公告时间对齐
│   ├── panel.py                #   事件面板组装与未来收益
│   ├── prices.py               #   日频行情下载 (东财/新浪, 带熔断降级)
│   ├── liquidity.py            #   ADV、波动、动量、流通市值 (入场前观测)
│   ├── fills.py                #   停牌/涨跌停导致的顺延成交
│   ├── tradability.py          #   停牌、涨跌停、T+1 与 ADV
│   ├── universe.py             #   历史时点股票池
│   ├── delisting.py            #   退市处理
│   ├── st_status.py            #   历史 ST 状态
│   ├── corporate_action.py     #   分红、送股、配股、拆分与复权
│   ├── eps_history.py          #   同花顺季度 EPS 历史 (SUE 用)
│   └── corrections.py          #   财报更正扫描
│
├── signals/                    # 基本面事件信号
│   ├── sue.py                  #   季节性随机游走 SUE
│   ├── earnings_growth.py      #   净利润/营收/EPS 同比、ROE 变化
│   ├── revenue_surprise.py     #   营收超预期
│   └── analyst_revision.py     #   分析师预测修正 (无数据源, 显式不可用)
│
├── factor/                     # 因子检验
│   ├── preprocess.py           #   去极值、标准化、walk-forward 切分
│   ├── neutralize.py           #   行业/市值中性化与横截面回归
│   ├── ic.py                   #   IC、RankIC 与 ICIR
│   ├── decay.py                #   不同预测期的 IC Decay
│   └── quantile.py             #   分层收益与单调性
│
├── portfolio/                  # 组合构建与风险指标
│   ├── construction.py         #   Long-only benchmark-relative 组合
│   ├── optimizer.py            #   风险平价与均值-方差 (v1.0 未用于结论)
│   └── risk.py                 #   Sharpe、MaxDD、TrackingError 与 IR
│
├── execution/                  # 交易成本模型
│   ├── transaction_cost.py     #   佣金和印花税
│   ├── slippage.py             #   滑点
│   └── market_impact.py        #   冲击成本与容量
│
├── backtest/
│   └── engine.py               #   月度事件队列回测与 Gross→Net 拆解
│
├── ml/                         # 机器学习实验
│   ├── linear.py               #   Ridge / 线性基线
│   ├── lightgbm.py             #   LightGBM / XGBoost
│   └── neural_net.py           #   MLP (v1.0 未进入结论)
│
├── research/
│   ├── config.py               #   v1.0 研究配置 (单一事实来源)
│   ├── 01_research_question.md #   研究问题与证伪条件
│   └── 02_akshare_mvp.md       #   AKShare 最小闭环计划
│
├── scripts/                    # 全流程入口 (run_all.py 一条命令跑完)
├── tests/                      # 201 个离线测试
├── results/                    # 表格、指标与图 (不提交)
├── report/                     # REPORT.md 研究报告
├── README.md
└── pyproject.toml
```

## 分阶段计划

> v1.0 状态：第 1–8 步已跑通并落盘结果；第 9 步完成了 Ridge 与 LightGBM 的样本外比较，
> 但 SHAP / PDP 解释与 MLP 未进入结论（见 Level 4 未通过项）。

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

- [x] 使用历史股票池并保留退市股票（股票池按披露日还原；但**数据源本身剔除了后来退市的公司**，属于已知限制）
- [x] 正确处理上市与退市日期
- [x] 使用真实公告时间，避免 look-ahead bias（91,598 条事件时序违规 0 条）
- [x] 正确处理复权、停牌和涨跌停（停牌/涨跌停顺延至首个可成交日）
- [x] 明确 ST / *ST 股票的处理规则（按披露日当时简称标记 3,038 条 ST 事件并全部剔除，最终面板 0 条）

### Level 2：研究设计成立

- [x] 完成 Signal 定义、RankIC、ICIR、分层收益和 IC Decay
- [x] 完成行业与市值中性化、横截面回归、分时期检验和机制检验

### Level 3：结果可以落到交易

- [x] 完成组合构建、换手率、佣金、印花税和滑点模拟
- [x] 加入流动性限制与市场冲击（10% ADV 参与率上限）
- [x] 报告 Gross vs Net、MaxDD 和 Capacity

### Level 4：机器学习有样本外增量

- [x] 样本外 RankIC 相对 SUE 基线改善（LightGBM 0.109 vs SUE -0.050）
- [x] 改善出现在多个年份（17 个样本外截面中 15 个为正）
- [ ] **改善并非由更高换手或微盘股集中造成 —— 未通过**：去掉 SUE 特征后表现几乎不变，说明增量不来自财报意外；且分年份收益在 -16% 与 +106% 之间摆动，低市值组净年化 13.7% 仍低于同组基准 31.5%

Level 4 最后一条未通过，因此**不能声称机器学习提供了基于财报事件的增量 Alpha**。

## 结果如何呈现

主报告 `report/REPORT.md` 给出以下内容：

- 数据覆盖范围、缺失情况、样本筛选过程与人工抽查
- SUE 的分布、覆盖率、RankIC、ICIR、分层收益和 IC Decay
- 中性化与横截面回归结果，包括系数、显著性和稳定性
- Gross 与 Net 收益、换手率、回撤、跟踪误差和容量估计
- 按年份、股票池、市场环境和关键参数拆分的稳健性结果
- 失败实验、异常结果及其排查过程
- 线性基线与机器学习模型在同一时间切分下的样本外比较

结果摘要把假设归入「得到支持」「证据不足」或「被当前实验否定」，并附上对应的统计结果和交易假设。项目是否完成取决于研究链条能否复现，`Sharpe > 2` 不在完成条件之内。
