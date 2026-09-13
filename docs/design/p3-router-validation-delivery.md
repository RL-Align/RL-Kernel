# P3 Router 验证基础设施交付文档

> **这是什么**：DSV4-Flash MoE Router（P3）的验证基础设施——验证阶梯、
> canonical 指纹、首差分定位、负向测试矩阵的设计与实现记录。
> **任务编号 T09 的含义**：见 `/workspace/p3-task-selection.md`（v22，§4-T09），
> 十人分工中"验证基础设施"一项；本文档即该项的交付说明。
> **怎么用**：审查人核对合同条款可跳读 §三 的 11 条决策；接手人从 §六 的
> 三项外部待办看起；复现测试看 §七。
> 交付日期：2026-09-09（开发 2026-09-07 ~ 09-09）
> 状态：范围内全部完成，等待上游 start kit 解锁剩余三项外部依赖
> 测试基线：**244 个测试收集，205 passed / 39 skipped（GPU 门控）**，CLI smoke 8/8 exit=0

---

## 一、背景与任务定位

### 1.1 P3 是什么

DSV4-Flash MoE 的 Router 层（合同 §1.1）：前 3 层用 Hash 路由（`tid2eid` 查表），
第 3 层起用 Learned 路由（correction-bias + stable Top-6）。P3 生产 `RoutePlan`，
供 P4（pack/transport）、P6（combine）、P7（cross-config）消费。

### 1.2 在十人分工里的位置

任务编号 T01–T10 各领一块（详见合同 `/workspace/p3-task-selection.md`）。
本任务（T09）是**验证基础设施**：别人写 router 算子，
我们写"怎么证明他们写对了"。核心交付四件事（合同 §4-T09 原文）：

1. 验证阶梯 L1/L2/L3a/L3b（+WS2 扩展）
2. 独立重写 naive Top-6，交叉检查 T01 golden
3. 按 `(absolute_layer, site, pass, event_index, global_token_id, rank)` 六元组定位首差分
4. 负向 fixture 矩阵（tie、XOR、bitflip、wrong-run、fallback、missing provenance、
   non-finite、selection-gradient）

### 1.3 开局约束：T01 未发布

开局核查（2026-09-08 确认，2026-09-09 复核）：`origin/main` 无任何 P3 代码、
无 `rl_engine/moe/` 目录、全分支搜不到 router 相关提交。T01 start kit 处于
`anchor_pending`（合同 §1.3）。

**合同 §5 明确允许并行**：T09 可先用自有 synthetic fixture 开发，不等 T01。
这决定了整个交付形态——所有依赖 golden/anchor 的部分做成"架子 + 自动生效的门控"，
T01 发布后零改动接入。

---

## 二、交付物清单

```
rl_engine/moe/
├── __init__.py                    10 行   包声明
├── naive_topk6.py                163 行   T09 自有 fixture：全序 Top-6（"独立重写"）
└── p3_verdicts.py                151 行   合同 §6 状态码冻结 + 优先级仲裁

rl_engine/moe/validation/
├── __init__.py                    12 行
├── report.py                      50 行   LadderReport + make_pass/make_fail（唯一构造点）
├── fingerprint.py                235 行   §2.4 canonical 序列化：semantic/artifact 双哈希
├── first_mismatch.py             199 行   六元组首差分定位 + owner/issue 归因表
├── comparison.py                 340 行   四级固定顺序比较器（§2.5）
├── ladder.py                     208 行   L1 repeat / L2 invariance / L3a oracle / L3b 双引擎
├── ws2.py                        134 行   rank 完整性 + cross-config partition/replica
├── paired_check.py               160 行   Torch paired check 架子（anchor_pending 门控）
└── synthetic_producer.py         152 行   合同 §5 授权的 seeded 确定性 producer

scripts/check_p3.py               245 行   CLI：--cases/--ladders/--seed/--rows/--json

tests/
├── test_naive_topk6.py           109 行    7 测试
├── test_p3_verdicts.py            90 行    6 测试
├── test_p3_comparison.py         203 行   16 测试
├── test_p3_validation_ladder.py  293 行   18 测试
├── test_p3_negative.py           382 行   21 测试
├── test_p3_ws2.py                210 行   12 测试
└── test_p3_paired_check.py       149 行   11+1skip 测试
```

共 13 个源文件 + 7 个测试文件，约 3500 行，244 个测试。

---

## 三、设计决策与理由（来龙去脉）

以下按"问题 → 决策 → 为什么不那样做"展开，每条对应合同的明确条款。

### D1：naive_topk6 独立重写，不依赖 T01

**合同要求**（§4-T09）："独立重写 naive Top-6，仅交叉检查 T01 golden"。

**实现**：全序排序语义 `(q 降序, logical_expert_id 升序)`，槽位保持排序原序；
FP32 精确比较；`cross_check_topk6` 输出首个 `(row, slot)` 差异。

**为什么写成独立模块**：交叉检查的价值恰恰在"独立性"——如果复用 T01 的
Top-6 实现来检查 T01 自己，同源错误会被系统性掩盖。所以哪怕 T01 的
`stable_topk6_device_abi.v1` 将来发布了，这个 naive 实现也不删，它是对拍基准。

**测试设计**：random / near-tie（ULP 阶梯）/ exact-tie（全同行必须输出 id 0..5）。
exact-tie 是关键——tie-break 政策 `(q 降序, id 升序)` 在全同行上退化为纯 id 升序，
这是最容易写错的地方。

### D2：状态码先行冻结（p3_verdicts.py）

**合同要求**（§6）：状态码表是 fail-closed 体系的骨架，"新增码只能追加，不得重排"，
且规定了三段 band：device 1–2 / provider 10–22 / runner 50–72。

**实现**：`P3Verdict` IntEnum + 设备可写集合校验 + `primary_verdict` 固定优先级仲裁。

**为什么第一个写它**：所有后续模块（比较器、阶梯、CLI exit code）都要引用状态码。
先冻结它，避免后面各模块各拿各的魔法数字。同时 `primary_verdict` 实现了合同 §6
"同一 case 多个错误按固定顺序取 primary"——这个优先级逻辑如果散落在各 runner 里，
必然各处不一致。

### D3：比较器四级固定顺序，身份门禁先行（comparison.py）

**合同要求**（§2.5）："比较顺序为 identity → discrete → score/weight → gradient；
任何 identity/schema/provenance/upstream verdict 缺失都停止比较"。

**实现**：`TraceComparator` 顺序走四个 stage，identity 漂移/缺失立即 halt——
后续再调任何 stage 方法直接抛 `RuntimeError`（防误用，而不是静默跳过）。

**为什么 halt 要抛异常**：调用方如果没检查 halt 状态继续喂梯度进来，说明调用方
逻辑有错。静默跳过会让"没比完"伪装成"比完了"，这正是 fail-closed 要防的。

**stop verdicts 集合**：identity/schema/upstream/provenance 类全在
`_STOP_VERDICTS` 里。后来（见 D8）把 `NON_FINITE` 也加了进去。

### D4：canonical 序列化与双哈希（fingerprint.py）

**合同要求**（§2.4）：semantic hash 按 per-token map、token 升序、padding 不进入；
artifact hash 按 `(case,config,rank)` 唯一、全部行含 padding + Envelope；
"不得只报 case hash"（必须能定位到 token 级）。

**实现**：
- `per_token_semantic_hashes` → dict[token, hash]，头部含 identity/layer/mode，
  行按 slot 升序序列化
- `route_semantic_hash` → case 级（token 升序折叠）
- `route_artifact_hash` → 全行含 padding + Envelope 字段
- 显式版本 `p3-t09-canonical.v1`

**为什么显式版本化**：合同 §8.1 规定 canonical hash 变化必须走 contract delta。
版本字符串嵌进 hash 头部，将来 T05 发布正式 schema 时原位替换，两个版本的
hash 天然不相等，不会被误当成"语义漂移"。

**为什么 padding 进 artifact 不进 semantic**：padding 合法地随 batch/pack 变化
（L2 的测试就依赖这一点），语义层必须剔除；但 L1 同配置 repeat 里 padding
也必须逐字节稳定，所以它在 artifact hash 里被审计。这就是"padding 只由同配置
artifact gate 审计"（§6 规则）的落地。

### D5：六元组定位与归因表（first_mismatch.py）

**合同要求**（§4-T09）："按 (absolute_layer, site, pass, event_index,
global_token_id, rank) 定位首差分；输出 owner/Issue/boundary/phase/artifact"。

**实现**：`MismatchKey` NamedTuple 即六元组；`_ATTRIBUTION` 静态表把 site 映射到
(owner, issue)：score→T02/#41、hash_lookup→T03/#42、topk→T01/#43、
selection→T04/#44、weight/handoff→T05、bwd→T06、tp_sp→T07、placement→T08。

**为什么用穷举静态表 + fail-closed**：site 集合是合同冻结的。未知 site 直接抛
`UnknownSiteError`——"refusing to guess attribution"。归因错误比没有归因更糟：
它会把 bug 派给错误的任务，浪费整个团队的时间。同理，backward pass 只允许
出现在梯度承载 site（score/bwd）上，出现在 topk 上说明 trace 本身就坏了。

### D6：四级阶梯的分工（ladder.py + report.py）

| 阶梯 | 比什么 | 用哪个哈希 | 抓什么错 |
|------|-------|-----------|---------|
| L1 | 同配置 repeat | artifact hash（全字节） | 非确定性：kernel 竞态、未初始化内存、atomics 顺序 |
| L2 | batch/pack/padding/launch 扰动 | per-token semantic hash | 布局泄漏：语义随物理布局变化 |
| L3a | candidate vs bit-defined oracle | 逐行 byte-exact | 算子实现错误 |
| L3b | recorded 双引擎（Megatron vs Miles） | 四级比较器 | 引擎间分歧 |

**L2 缺 token → INCOMPLETE_ARTIFACT(13)、多 token → AMBIGUOUS_GLOBAL_TOKEN_MAPPING(63)**
——直接对应合同 §2.4 的 runner 规则。

**report.py 的由来（自审产物）**：最初 `LadderReport` 和 `_fail`/`_pass` 写在
ladder.py 里，ws2.py 和 paired_check.py 跨模块导入下划线私有函数——违反封装。
重构抽出 `report.py`，公开 `make_fail`/`make_pass`，依赖方向变为单向：
`CLI → ladder/ws2/paired_check → report → 底层原语`。

### D7：WS2 的两种 token 所有权模式（ws2.py）

**合同要求**（§2.4）：tp 为 replica 或 partition_by_sequence、dp/cp 为 partition、
pp 为 replica_by_layer、ep 不是 token 维度；缺失→MISSING_RANK/INCOMPLETE、
越权或重复→AMBIGUOUS。

**实现**：`check_rank_completeness`（缺 rank→20，重复 rank→22 stale）+
`run_ws2_cross_config`（partition：token 恰好出现一次；replica：可重复但每个
载体的 hash 必须与 base 一致，首个 `(token, rank)` 定位）。

**为什么 replica 不去重比较**：TP 下同一 token 的多份拷贝如果只比一份，
另一份坏了就漏检。逐载体比对才能抓住"某个 rank 单独漂移"。

### D8：non-finite 前置门（自审补齐的关键缺口）

**合同要求**（§2.5）："forward active 行的 z'、s、q、a、Z、p、w 非有限为
NON_FINITE"；§6："非 PASS 的 non-finite fail-closed"。

**最初遗漏**：第一轮交付的比较器完全没有 non-finite 检查路径。后果很具体：
NaN 进 byte gate，`NaN != NaN` 会被误报成 `ROUTE_WEIGHT_BYTES_MISMATCH(51)`，
归因派给 T02 的"字节不等"——但真实缺陷是某处算出了 NaN，两者的修复路径完全不同。

**修复**：`_nonfinite_gate` 挂在 stage 3/4 的字节比较**之前**，任一侧 active 值
非有限即 `NON_FINITE(1)` 且加入 `_STOP_VERDICTS` 停走。四条边界用测试钉死：
① NaN 不被误报为 bytes mismatch；② padding 行非有限不判（§2.5 只查 active，
有 mask 的用例验证豁免）；③ P3 自身(1) 与 upstream(18) 两个 band 不混淆；
④ 梯度侧 dz 非有限归因 T06。

### D9：XOR 负向（同一次自审补齐）

**合同要求**（§2.5）："Hash/Learned 是按 (absolute_layer, router_mode) 的 XOR"
——一个 layer 在同一时刻只能是一种模式，模式互斥。

**最初状态**：机制存在（identity gate 的 `router_mode` 字段、semantic hash 头部
都含 mode），但没有显式负向用例。合同 §4-T09 验收行明确列了 XOR。

**补齐**：两个用例——同一 layer 两侧 trace 模式不一致必须 halt 在 identity gate
（`IDENTITY_DRIFT`，到不了数值阶段）；仅翻转 mode 字段必须改变 per-token
semantic hash（否则 XOR 约束形同虚设），L2 以 59 拦截。

### D10：paired_check 架子与 anchor_pending 纪律

**合同要求**（§2.5）：Torch 原始参考必须在每个正式 golden 上运行 paired check
并记录诊断；"其差异不能覆盖 strict verdict，缺证据为 MISSING_PROVENANCE(67)"。

**实现**：manifest 查找 `fixtures/p3/manifest.json`，不存在→空列表+
`anchor_pending`。四条规则：paired diff 仅诊断永不翻转 verdict；缺执行证据→67
（fail-closed，不是绿）；Torch crash 也是缺证据不是 pass；gate 要求每个正式
golden 都有证据槽位。T01 自带的 paired record 可短路。真实 manifest 集成测试
用 `skipif` 门控，T01 发布后自动生效。

**为什么不猜测 manifest 路径**：合同 §1.3 "we look it up, we never guess"。
一个写死的猜测路径如果恰好命中过时的缓存文件，会静默给出错误的"证据完整"结论。

### D11：CLI 的 exit code 语义与输出分流

**实现**：`check_p3.py` exit code = 首个失败 verdict 的数值码（全过为 0）——
CI 里可以直接 `echo $?` 区分失败类别。`--json` 模式下结构化报告走 stdout、
人读摘要走 stderr，互不污染（管道 `| jq` 不再被摘要行打断）。

**修掉的历史包袱**：初版在函数体内用 `__import__(..., fromlist=[...])` 动态
导入 `MismatchKey`——纯 hack。清理为顶部静态导入。

---

## 四、测试矩阵（244 个测试的设计逻辑）

| 文件 | 数量 | 钉死什么 |
|------|-----|---------|
| test_naive_topk6 | 7 | 全序语义、tie-break、交叉检查定位 |
| test_p3_verdicts | 6 | 状态码 band、追加不重排、primary 优先级 |
| test_p3_comparison | 16 | 四级顺序、halt 语义、±0.0 区分、padding 排除 |
| test_p3_validation_ladder | 18 | 四阶梯各自的 pass/fail 边界、CLI 参数化 |
| test_p3_negative | 21 | 单一缺陷注入→精确 verdict（见下） |
| test_p3_ws2 | 12 | partition/replica、缺/重 rank、(token,rank) 定位 |
| test_p3_paired_check | 11+1skip | 诊断不翻转、缺证据 67、manifest 容错 |

**负向矩阵的坚持**：每个用例只注入**单一**缺陷。多缺陷混注会让"verdict 正确"
变成巧合（可能被另一个缺陷的 verdict 掩盖），单缺陷才能证明归因逻辑本身正确。
另有控制组反向钉死：行序重排/padding 增删/Envelope 变化**必须**判 invariant
（防误报——验证基础设施自己的假阳性同样致命）。

39 个 skipped 全部是 GPU 门控用例（本机无 GPU），与逻辑无关。

**负向矩阵明细**（缺陷 → 期望 verdict → 拦截层）：

| 注入缺陷 | verdict | 拦截层 |
|---------|---------|--------|
| tie-break 违反（降序 id 打破平票） | 55/56 | cross_check / L3a |
| Hash/Learned 模式互斥违反（XOR） | 10 halt | identity gate |
| 仅翻转 router_mode 字段 | 59 | L2 semantic hash |
| weight bitflip（active 行） | 51 | L3a |
| score bitflip | 52 | L3a |
| semantic 字段改动 | 59 | L2 |
| artifact 字段改动（含 padding） | 60 | L1 |
| stale run/attempt metadata | 60 | L1 |
| missing provenance（identity 字段缺失） | 67 halt | identity gate |
| identity drift（checkpoint/weight 变化） | 10 halt | 优先于后续阶段 |
| L2 丢 token | 13 | L2 |
| L2 幽灵 token | 63 | L2 |
| 禁止的 silent fallback 标志 | 66 halt | provenance |
| selection 梯度泄漏 | 53 | L3b（归因 T06） |
| active 值 NaN/Inf | 1 halt | non-finite 前置门 |
| padding 增删试图掩盖 Core 改动 | 59 | L2（控制组同时钉死合法 padding 不误报） |

---

## 五、自审与迭代记录

这个交付经过一轮显式自审（"当前所有的修改全部结束了？自己审查一下代码"），
发现并修复了三个结构问题、两个合同覆盖缺口：

| # | 类型 | 问题 | 处置 |
|---|------|------|------|
| 1 | 结构 | `LadderReport`/`_fail`/`_pass` 定义在 ladder.py，ws2/paired_check 跨模块导入私有函数 | 抽出 report.py，公开 make_pass/make_fail |
| 2 | 结构 | CLI 里 `__import__` 动态导入 hack | 改顶部静态导入 |
| 3 | 结构 | `--json` 模式摘要行污染 stdout，管道解析崩 | JSON 走 stdout、摘要走 stderr |
| 4 | 覆盖 | 合同负向清单里的 XOR 无显式用例 | +2 测试（D9） |
| 5 | 覆盖 | 合同负向清单里的 non-finite 完全没有比较器路径 | `_nonfinite_gate` + NON_FINITE 入停走集合，+4 测试（D8） |

这次自审的价值在于第 5 条：不看合同原文逐字核对，"NaN 会被误报成 bytes
mismatch"这种归因错误永远不会被测试暴露——测试全绿，但绿的是错误的 verdict。

---

## 六、边界与未做之事（防跑偏声明）

**T09 只做验证基础设施，不写 router 算子**（那是 T02–T04/T06 的活）。
具体边界：

- `naive_topk6.py` 是对拍基准，不是生产实现——不进任何 provider 路径
- `fingerprint.py` 的 `p3-t09-canonical.v1` 是占位 canonical 序列化，
  T05 发布正式 schema 后**原位替换**（合同 §4-T05："禁止维护第二份字段表"）
- `synthetic_producer.py` 只用 seeded 确定性数据，符合 §7 DoD
  "仅使用 synthetic/sanitized fixture"
- 不改 `rl_engine/kernels/gtest/` 等他人代码，只读复用模式

**三项外部依赖待办**（均被 T01 阻塞，合同允许的 anchor_pending 状态）：

1. T01 发布 golden manifest 后，paired_check 的 skipif 集成测试自动生效
2. T05 发布正式 schema 后，替换 fingerprint.py 的序列化并升版本号
3. 认领公示（合同 §8.3 群内回复格式）需人工确认

**已知邻近风险**：`origin/dsv4-p5-dev` 分支的 P5 starter 已提交
`rl_engine/moe/{contract,oracle,fixtures,...}.py`，与 T09 的文件零重叠但共享
`rl_engine/moe/` 目录。T09 的 `__init__.py` 只有 10 行包声明、不导出任何符号，
合并时冲突风险极低；若 P5 先进 main，T09 侧只需保留对方 `__init__.py`。

---

## 七、复现命令

```bash
# 全部 T09 测试（含 naive_topk6 与全部 p3 相关）
cd /workspace/RL-Kernel
python -m pytest tests/ -q -k "topk or p3"
# → 205 passed, 39 skipped in ~17s

# 单文件
python -m pytest tests/test_p3_negative.py -q          # 21 passed
python -m pytest tests/test_p3_validation_ladder.py -q  # 18 passed

# CLI 冒烟（2 case × 4 ladder）
python scripts/check_p3.py --cases smoke
# → check_p3: cases=2 ladders=L1,L2,L3a,L3b passed=8/8 exit=0

# CLI 结构化输出（stdout 纯 JSON，可管道）
python scripts/check_p3.py --cases smoke --json | jq '.[0].ladder'

# exit code 即首个失败 verdict 码（CI 可用 $? 区分失败类别）
python scripts/check_p3.py --ladders L1; echo $?
```

---

## 八、模块依赖图

```
scripts/check_p3.py (CLI)
        │
        ▼
ladder.py ── ws2.py ── paired_check.py        ← 三个 runner 层
        │            │
        └────┬───────┘
             ▼
          report.py                            ← LadderReport 唯一构造点
             │
             ▼
comparison.py (TraceComparator, 四级)          ← 比较原语
  first_mismatch.py (六元组+归因)               ← 定位原语
  fingerprint.py (semantic/artifact hash)      ← 序列化原语
             │
             ▼
        p3_verdicts.py                         ← 状态码冻结（最底层，无依赖）

synthetic_producer.py → ladder/ws2 (fixture)
naive_topk6.py        → 独立，仅被测试与交叉检查引用
```

无循环依赖；每层只向下依赖。`p3_verdicts.py` 位于最底层且被所有层引用——
这正是 D2"先行冻结"的 structural 体现。

---

## 九、合同验收项 → 代码 → 测试 对照表（review 导航）

review 时按此表逐项核对，左列即合同 §4-T09 验收行的原文要点。

| 合同验收项 | 代码位置 | 测试位置 |
|-----------|---------|---------|
| 阶梯 L1 repeat | `ladder.py:48 run_l1_repeat` | `test_p3_validation_ladder.py` |
| 阶梯 L2 invariance | `ladder.py:75 run_l2_invariance` | 同上 |
| 阶梯 L3a oracle | `ladder.py:116 run_l3a_oracle` | 同上 |
| 阶梯 L3b Megatron-vs-Miles | `ladder.py:172 run_l3b_dual_engine` | 同上 |
| WS2 rank 完整性（缺 20/重 22） | `ws2.py:43 check_rank_completeness` | `test_p3_ws2.py` |
| WS2 cross-config partition/replica | `ws2.py:73 run_ws2_cross_config` | 同上 |
| 独立重写 naive Top-6 | `naive_topk6.py:95 naive_topk6` | `test_naive_topk6.py` |
| 交叉检查首差分 (row,slot) | `naive_topk6.py:126 cross_check_topk6` | 同上 |
| 状态码冻结 + primary 仲裁 | `p3_verdicts.py:107 primary_verdict` | `test_p3_verdicts.py` |
| canonical semantic/artifact 双哈希 | `fingerprint.py:148/169/217` | `test_p3_validation_ladder.py` |
| 六元组首差分 + 归因 | `first_mismatch.py:125 first_mismatch` | `test_p3_comparison.py` |
| 四级顺序比较器 | `comparison.py:136 TraceComparator` | 同上 |
| non-finite 前置门 | `comparison.py:233 _nonfinite_gate` | `test_p3_negative.py` |
| paired check（诊断不翻转/缺证据 67） | `paired_check.py:92 run_paired_check` | `test_p3_paired_check.py` |
| anchor_pending manifest 门控 | `paired_check.py:67 load_golden_manifest` | 同上 |
| 负向矩阵（8 类缺陷→精确 verdict） | —（注入在测试内构造） | `test_p3_negative.py` |

---

## 十、公开 API 一览（按模块）

**`naive_topk6.py`**：`naive_topk6(q) -> (ids, weights)` ·
`cross_check_topk6(candidate, expected) -> 首差 (row, slot) | None`

**`p3_verdicts.py`**：`P3Verdict`（IntEnum）· `primary_verdict(list) -> P3Verdict | None` ·
`is_valid_device_status(int)` · `classify_writable_band(int)`

**`fingerprint.py`**：`route_semantic_hash(rows, identity, layer) -> str` ·
`per_token_semantic_hashes(...) -> dict[token, hash]` ·
`route_artifact_hash(artifact) -> str`；数据类 `RouteRow / RouteIdentity /
EnvelopeFields / Artifact`

**`first_mismatch.py`**：`first_mismatch(lhs_events, rhs_events) -> FirstMismatch | None`；
`MismatchKey`（六元组 NamedTuple）；异常 `UnknownSiteError`

**`comparison.py`**：`TraceComparator`（`check_identity / check_discrete /
check_score_weight / check_gradient / report`）；独立函数 `tensor_byte_exact /
discrete_equal`

**`ladder.py`**：`run_l1_repeat / run_l2_invariance / run_l3a_oracle /
run_l3b_dual_engine`，全部返回 `LadderReport`

**`ws2.py`**：`check_rank_completeness / run_ws2_cross_config`

**`paired_check.py`**：`load_golden_manifest / run_paired_check /
paired_gate_for_goldens`

**`report.py`**：`LadderReport` · `make_pass / make_fail`（唯一构造点）

**CLI `check_p3.py`**：参数 `--cases/--ladders/--seed/--rows/--json`；
exit code = 首个失败 verdict 码（全过 0）

