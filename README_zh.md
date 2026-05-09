# CodePipe

[English](README.md) | 中文版

**多语言、本地优先的确定性流水线 Coding Agent。**

CodePipe 是一个基于 Agentless（ICSE 2025）理念构建的 CLI 编程代理：LLM 只做分类和生成，确定性代码做决策和验证。不同于 ReAct 循环代理（Claude Code、Cursor），CodePipe 采用固定的 5 专家流水线，专为本地小模型（8B–30B）优化设计。

## 为什么做 CodePipe？

ReAct 循环代理需要强推理模型来决策"下一步调用什么工具"。本地 8B 模型容易陷入死循环、幻觉工具调用、用同样方式反复失败。CodePipe 将决策循环替换为确定性流水线——LLM 只出现两次：一次分类任务，一次生成补丁。

## 架构

```
用户输入 → Gate → Locator → Generator → Verifier → 输出
              ↑         ↑          ↑          ↑
          LLM 调用   BM25+AST   LLM 调用   ast+pytest
```

| 专家 | 职责 | 调 LLM？ |
|------|------|----------|
| **Gate** | 将任务分类为 7 种类型 | 是（单次调用） |
| **Locator** | BM25 + AST 调用图代码搜索 | 否 |
| **Generator** | CREATE/EDIT 模式 + SEARCH/REPLACE 块 | 是 |
| **Verifier** | L1 语法检查 + L2 pytest 测试执行 | 否 |
| **Debugger** | sys.settrace 运行时变量捕获 | 否 |
| **Reviewer** | 修复后需求对齐审查 | 是 |

## 核心特性

- **多驱动兼容** — config.yaml 一键热切换 DeepSeek API / Ollama / 任意 OpenAI 兼容接口
- **两阶段定位** — BM25 关键词召回 + AST 调用图展开，不调 LLM，<3 秒
- **模糊补丁匹配** — SEARCH/REPLACE 块格式，difflib 降级容错，85% 相似度阈值，容忍缩进漂移
- **双层验证护栏** — L1: ast.parse 语法检查 → L2: pytest 测试执行，区分 IMPORT_ERROR 和代码逻辑错误
- **Git 原子状态机** — 每次任务前 git 快照，失败时 `git reset --hard` 原子回滚，不留烂代码
- **防死锁重试** — 记录每次失败的尝试，逐级增强警告注入，禁止 LLM 重复相同方案
- **Reflexion 经验进化** — 失败→成功模式持久化到 REFLECTION.md，下次任务作为 Few-shot 注入
- **Top-K 并发采样** — asyncio 并发生成 K 个候选补丁，首个通过即胜出
- **数据飞轮** — 收集 (instruction, context, output) 三元组到 dataset.jsonl，为 LoRA 微调储备数据
- **Docker 沙盒** — 容器隔离的 L2 测试执行，工作区只读挂载，宿主机安全
- **TDBR 测试复现** — 先写失败测试复现 Bug，再修复代码让测试变绿
- **调用图语义切片** — 基于 AST 的上游（Def-Use）+ 下游（Callers）依赖上下文提取

## 快速开始

```bash
pip install -e ".[dev]"
# 编辑 config.yaml 配置 API key
pytest tests/ -q                    # 179 个测试应全部通过
```

```bash
# 与配置的 LLM 对话
python cli.py chat "你好"

# 列出可用驱动
python cli.py providers
```

## 配置

```yaml
# config.yaml
active: deepseek  # 或 ollama

providers:
  deepseek:
    base_url: "https://api.deepseek.com/v1"
    api_key: "${DEEPSEEK_API_KEY}"
    model: "deepseek-chat"
  ollama:
    base_url: "http://localhost:11434/v1"
    api_key: "ollama"
    model: "qwen3:8b"
```

或通过环境变量：`CODEPIPE_BASE_URL`, `CODEPIPE_API_KEY`, `CODEPIPE_MODEL`。

## 设计哲学——七条红线

1. **禁止庞大生态框架** — 不引入 LangChain、LlamaIndex、向量数据库
2. **禁止硬编码 Provider** — LLMClient 抽象层，构造参数接收任意 base_url
3. **禁止多 Agent 自由路由** — 不用 AutoGen、CrewAI，不允模型自我决定下一步
4. **确定性流水线** — Input → Gate → Locator → Generator → Verifier → Output
5. **TDD 强制** — 所有核心逻辑先写 pytest 测试，179 测试覆盖全部 7 个阶段
6. **LLM 只做分类和生成** — 流程控制 100% 确定性代码
7. **数据不出网** — 全部本地运行，模型跑本地，搜索可选自部署 SearXNG

## 项目结构

```
codepipe/
├── cli.py                     # Typer 入口
├── config.yaml                # 多驱动配置
├── core/
│   ├── llm_client.py          # 统一 LLM 驱动
│   ├── orchestrator.py        # 流水线编排 + Git 状态机
│   ├── generator.py           # SEARCH/REPLACE + 模糊匹配
│   ├── topk_sampler.py        # 并发 K 候选生成
│   ├── tdbr_reproducer.py     # 测试驱动 Bug 复现
│   ├── data_flywheel.py       # LoRA 训练数据收集
│   ├── docker_sandbox.py      # 容器隔离测试运行
│   ├── locator/
│   │   ├── bm25_scorer.py     # BM25 文件评分
│   │   ├── ast_extractor.py   # 多语言 AST 提取
│   │   ├── call_slicer.py     # 调用图上下文切片
│   │   └── locator.py         # 组合两阶段定位
│   └── verifier/
│       └── verifier.py        # L1 语法 + L2 测试验证
├── memory/
│   └── reflection.py          # REFLECTION.md 持久化
└── tests/                     # 179 个测试，覆盖 7 个阶段
```

## 各阶段详情

| 阶段 | 内容 | 测试数 |
|------|------|--------|
| Phase 1 | LLMClient 多驱动统一层 | 24 |
| Phase 2 | Locator BM25 + AST 上下文裁剪 | 25 |
| Phase 3 | Generator SEARCH/REPLACE + difflib 模糊匹配 | 38 |
| Phase 4 | Verifier L1/L2 + Git 状态机 + 防死锁重试 | 50 |
| Phase 5 | Reflexion 经验进化（REFLECTION.md） | 18 |
| Phase 6 | Top-K 采样 + 数据飞轮 + Docker 沙盒 | 10 |
| Phase 7 | TDBR 测试复现 + 调用图语义切片 | 14 |

## 理论基础

| 论文 / 项目 | 来源 | 在 CodePipe 中的应用 |
|-------------|------|---------------------|
| **Agentless** | Xia et al., ICSE 2025 | 整体架构：确定性流水线优于复杂 agent |
| **CodeCompass** | arXiv:2602.20048, 2026 | AST 调用图定位，G3 任务 99.4% |
| **Debug2Fix** | Microsoft, ICML 2026 | 弱模型 + 调试器 > 强模型裸跑 |
| **LLMCompiler** | ICML 2024 | DAG 任务并行调度 |
| **Reflexion** | NeurIPS 2023 | 失败模式持久化 |

## 借鉴的开源项目

- **Claude Code** (Anthropic) — CLAUDE.md → KWCODE.md，Checkpoint 机制
- **OpenHands V1** — Agent delegation 任务分解、Context Condensation
- **SearXNG** — 零 API key 本地搜索引擎
- **rank-bm25 / tree-sitter** — BM25+ 算法、多语言 AST 解析

## License

MIT
