# A 股基本面事件 Alpha 研究项目

**A-share Fundamental Event Alpha: Point-in-Time Signals, Trading Frictions and Machine Learning**

一个试图完整复现一次真实 A 股量化研究流程的项目。它的目标**不是**做一个“看起来收益很高”的策略 Demo，也不是单纯复现一个 PEAD 因子，而是回答：

> 一个量化信号应该**如何提出、如何验证、如何被质疑**，并最终判断它是否具有**可交易价值**。

---

## 项目定位

以 **财报公告后的价格漂移（PEAD）** 为切入口，跑通完整的量化研究能力链：

```
金融问题 → 研究假设 → 真实数据 → Alpha Signal → 统计验证
     → 组合构建 → 交易约束 → 交易成本 → 风险分析 → 归因
     → 传统因子 → 机器学习 → 增量 Alpha 检验
```

最终定位是一个 **Quant Research Case Study**，而不是 **Trading Strategy Demo**。

> 我们想证明的不是“我会 Python、LightGBM、PyTorch”，而是“我知道一个量化信号应该如何被严谨地研究”。

---

## 核心研究问题

> **A 股市场在财报信息公布后，是否存在持续性的价格反应？这种现象在消除未来函数（look-ahead bias）、幸存者偏差（survivorship bias），并加入真实交易约束之后，是否仍然存在？**

第一阶段从 **PEAD** 入手，定义 earnings surprise：

```
SUE_{i,t} = (ActualEPS - ExpectedEPS) / σ(EarningsSurprise)
```

核心假设：

```
SUE ↑  ⇒  FutureReturn ↑
```

即：财报表现越超预期，公告后的股票在未来若干交易日是否继续表现更好？

若暂时缺乏分析师预期数据，可用以下早期代理：

- 净利润同比变化
- 营收同比变化
- EPS 同比变化
- ROE 变化
- 毛利率变化
- 业绩预告超预期 proxy

---

## 与普通学生量化项目的区别

**普通项目：**

```
下载股票 → 算一个因子 → 分组 → 回测 → Sharpe 很高
```

**本项目：**

```
历史真实股票池 → 真实公告时间 → 构造基本面事件信号 → 检查是否真的可交易
→ 因子有效性检验 → 风险暴露控制 → 组合构建 → 交易模拟
→ 交易成本 → 净收益 → 机制分析 → 稳健性与失败分析 → 机器学习扩展
```

---

## 项目结构

```
a-share-fundamental-alpha/
│
├── data/                       # Point-in-Time 数据环境
│   ├── universe.py             #   历史时点真实股票池（含退市/被收购/ST）
│   ├── financials.py           #   财报数据
│   ├── announcement.py         #   真实公告时间对齐（避免 look-ahead）
│   ├── corporate_action.py     #   分红/送股/配股/拆分/复权
│   ├── delisting.py            #   退市处理（避免幸存者偏差）
│   └── tradability.py          #   停牌/涨跌停/T+1/ADV 等可交易性
│
├── signals/                    # Alpha Signal 构造
│   ├── sue.py                  #   Standardized Unexpected Earnings
│   ├── earnings_growth.py      #   净利润/营收同比变化
│   ├── revenue_surprise.py     #   营收超预期
│   └── analyst_revision.py     #   分析师业绩修正
│
├── factor/                     # 统计验证
│   ├── preprocess.py           #   数据清洗/填充/去极值/标准化
│   ├── neutralize.py           #   行业/市值中性化
│   ├── ic.py                   #   IC / RankIC / ICIR
│   ├── decay.py                #   IC Decay（不同预测期）
│   └── quantile.py             #   Q1–Q10 分层收益
│
├── portfolio/                  # 组合构建
│   ├── construction.py         #   Long-only benchmark-relative
│   ├── optimizer.py            #   权重/暴露/换手约束优化
│   └── risk.py                 #   Sharpe/MaxDD/TrackingError/IR
│
├── execution/                  # 真实交易成本
│   ├── transaction_cost.py     #   佣金 + 印花税
│   ├── slippage.py             #   滑点
│   └── market_impact.py        #   冲击成本（OrderSize/ADV）
│
├── backtest/
│   └── engine.py               # 事件/组合回测引擎
│
├── ml/                         # 机器学习扩展
│   ├── linear.py               #   Baseline：线性模型
│   ├── lightgbm.py             #   Tree Model：LightGBM / XGBoost
│   └── neural_net.py           #   NN：MLP
│
├── research/                   # 研究过程记录（自下而上推进）
│   └── 01_research_question.md #   ★ 当前进度：研究问题
│
├── tests/
├── report/
│   └── research_report.pdf
│
├── README.md
└── pyproject.toml
```

---

## 项目主线（分阶段推进）

每一阶段都会产出可验证的成果，并作为下一步的输入。**不要跨阶段**。

### 1. 金融问题 & 研究假设
- 明确定义要研究的现象（PEAD）
- 写清 economic motivation、核心假设、可检验的指标
- 说明“为什么这个信号可能有效”（信息反应不足 / 关注度 / 套利限制……）

### 2. 真实数据（Point-in-Time）
- 构造历史时点真实股票池（含后来退市/被收购/破产/ST 的公司）→ **避免幸存者偏差**
- 按*真实公告时间*进入模型，保证 `InformationTime < TradingTime` → **避免未来函数**
- 处理上市/退市/停牌/涨跌停/复权/T+1/印花税，并纳入 ADV 估算容量

### 3. Alpha Signal
- 构造 SUE / 增长类基本面信号，定义明确、可复现

### 4. 统计验证
- IC / RankIC / ICIR、Q1–Q10 分层收益（观察**单调性**）、IC Decay（1–60 天）
- 行业/市值中性化 + 横截面回归，观察 `β₁(SUE)` 是否仍显著
- 机制分析：按市值/流动性/波动率/分析师覆盖拆分

### 5. 组合构建
- Long-only benchmark-relative portfolio
- 单票权重/行业/市值/换手/流动性约束

### 6. 交易约束 & 交易成本
- 佣金 + 印花税 + 滑点 + MarketImpact
- 报告 Gross Alpha → Net Alpha 的拆解（项目的**核心亮点**之一）

### 7. 风险分析 & 归因
- 报告 AnnualReturn / Sharpe / MaxDD / Turnover / TrackingError / InformationRatio

### 8. 稳健性 & 失败分析
- 主动攻击策略：换股票池（沪深300/中证500/中证1000）、子时期、市场环境、持有期、成本假设、信号定义
- 若参数略变结果就消失 ⇒ 警惕过拟合

### 9. 传统因子 → 机器学习 → 增量 Alpha 检验
- 特征：`[SUE, RevenueGrowth, ProfitGrowth, Momentum, Volatility, Liquidity, Size, Industry, MarketRegime]`
- 模型：Linear → LightGBM → MLP（不用 LSTM 直接预测价格）
- 评价：RankIC / ICIR / Net Sharpe / Turnover / MaxDD / Capacity
- 用 SHAP / PDP 解释模型学到的 interaction（如 `SUE × Liquidity`）

---

## 验收标准

### Level 1：数据正确 ⚠️（不通过则后续一律作废）
- [ ] 使用历史真实股票池，包含已退市股票
- [ ] 正确处理上市/退市日期
- [ ] 使用真实公告时间，无 look-ahead bias
- [ ] 正确处理复权、停牌、涨跌停
- [ ] 明确 ST / *ST 处理规则

### Level 2：研究正确
- [ ] Signal 定义 / RankIC / ICIR / Q1–Q10 分层 / IC Decay
- [ ] 行业 & 市值中性化 / 横截面回归 / Subperiod test / Mechanism test

### Level 3：策略真实
- [ ] Portfolio construction / Turnover / Commission / Stamp Tax / Slippage
- [ ] Liquidity constraint / Market Impact / Gross vs Net / Max DD / Capacity

### Level 4：ML 有真正增量
- [ ] Out-of-sample RankIC / ICIR / Net Sharpe 提升
- [ ] 提升在多个年份存在，且非来自更高换手或 Micro-Cap 集中
- [ ] 扣除交易成本后仍然存在

---

## 项目成功标准

成功**不等于** `Sharpe > 2`。即使最终发现：

> PEAD 在成本前明显，但加入真实交易成本后只剩很弱的收益

项目仍然可以是成功的。真正重要的是：

- 数据正确
- 研究假设明确
- 结果可解释
- 偏差被控制
- 策略可交易性被检验
- 失败原因被分析

---

## 状态

当前进度：**第 1 阶段 —— 研究问题**。详见 `research/01_research_question.md`。

> ⚠️ 里程碑式推进中，尚未进入数据阶段。在 `Level 1`（数据正确）达成之前，任何回测结果都不采信。

---

*项目文档基于求职导向的量化研究框架整理，逐步推进。*
