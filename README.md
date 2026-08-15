# 探索实验|prompt工程有用吗

作为一名初学者，我希望通过一个包含30道测试题的小型非正式实验，初步探究提示工程（Prompt Engineering）的有效性，
并观察其在轻量级模型与最先进（SOTA）模型上的效果差异。

本次实验旨在把握大致趋势，而非进行严格的学术论证。

## prompt工程解释

 prompt工程目标：是通过精心设计的输入文本，引导模型输出符合预期的高质量结果。

 一个高质量的 Prompt 通常包含五个核心要素：角色（Role）、任务（Task）、上下文（Context）、约束（Constraints）和格式（Format）。这五个要素共同构成了 Prompt 的"黄金结构"。

 Few-Shot Learning 是 Prompt Engineering 中最强大的技术之一。通过在 Prompt 中提供少量输入-输出示例，你可以让模型快速理解任务模式，而无需重新训练模型参数。的核心原理是 In-Context Learning（上下文学习）。

 Chain of Thought（CoT）Prompting 的核心思想是：引导模型将复杂问题拆解为多个中间推理步骤，而不直接跳到答案。两种实现方式：Zero-Shot CoT（只需在 Prompt 末尾加上"请逐步思考"）和 Few-Shot CoT（在示例中展示完整的推理过程）

 ReAct（Reasoning + Acting）是一种将推理和行动交替进行的 Prompt 设计模式。与纯 CoT 不同，ReAct 让模型在推理过程中可以调用外部工具（如搜索引擎、API、数据库），然后根据工具返回的结果继续推理。

 System Prompt是对话开始前注入模型的"系统级指令"，它定义了模型的行为边界、角色定位和输出风格。

 自洽性的工作流程是：对同一个问题，使用思维链提示生成 N 条（通常 5-40 条）不同的推理链，然后从这些推理链的最终答案中通过投票选出出现频率最高的那个。

 结构化输出：直接输出JSON 对象、XML 文档、CSV 表格等，以便后续程序可以直接解析和处理。

 Tree of Thoughts（思维树）：核心思想：不是只沿着一条推理路径走到底，而是生成多个可能的推理步骤，评估每个步骤的质量，选择最有希望的路径继续深入。

  本实验是对其中Few-Shot Learning，Zero-Shot CoT，Few-Shot CoT，Zero-Shot CoT + 自洽性，Few-Shot CoT + 自洽性进行可以量化的验证。

 ### 2026年业界总结prompt的最佳实践

推理任务必用 CoT
任何涉及数学、逻辑、推理的问题，都应该使用 Chain-of-Thought。Zero-Shot CoT（加一句"让我们一步步思考"）是最低成本高回报的优化。

高准确度需求用 Self-Consistency
当需要 90%+ 的准确率时，Self-Consistency（k=5）是性价比最高的选择。

需要外部能力用 ReAct
搜索、计算、API 调用等场景，ReAct 框架是标准模式。

复杂规划用 Tree of Thoughts
创意写作、多步规划、策略制定等需要多路径探索的场景。

结构化输出用 JSON Schema
数据提取、API 响应、格式转换等场景，强制 JSON Schema 约束。

Prompt 需要版本管理和 A/B 测试
把 Prompt 当代码管理：版本控制、测试用例、性能监控。

持续评估和优化
建立评估集（Evaluation Suite），每次 Prompt 变更后自动回归测试。



## 实验方法
策略:
E1 Zero-shot 基线          (temp=0.0, 1次)
E2 Zero-shot CoT           (temp=0.0, 1次)
E3 Few-shot CoT            (temp=0.0, 1次)
E4 Zero-shot CoT + 自洽性  (temp=0.7, 5次投票)
E5 Few-shot CoT + 自洽性   (temp=0.7, 5次投票)

数据集：
GSM8K 测试集前30道

模型：
qwen2.5-3b（本地部署）
deepseek-v4-flash


## 环境准备
python 3.8+，安装依赖：
    pip install -r requirements.txt
运行 API 模型需设置环境变量（PowerShell）：
    $env:DEEPSEEK_API_KEY = "sk-xxx"


## 运行
    python experiment.py --limit 30

## 结果

### 准确率总表（正确数 / 30 题）

| 模型 | E1 Zero-shot | E2 CoT | E3 Few-shot CoT | E4 CoT+SC | E5 Few-shot+SC |
|:--|:--:|:--:|:--:|:--:|:--:|
| **本地 Qwen2.5-3B** | 20.0% | 66.7% | 66.7% | **83.3%** | 73.3% |
| **DeepSeek-V4-Flash** | 93.3% | 93.3% | 93.3% | **96.7%** | **96.7%** |

### 投票集中度（E4/E5 各采样 5 次，众数答案的平均票数）

| 模型 | E4 | E5 |
|:--|:--:|:--:|
| 本地 Qwen2.5-3B | 3.63/5 | 3.60/5 |
| DeepSeek-V4-Flash | 4.60/5 | 4.77/5 |

说明：Qwen 在温度 0.7 下约 1/3 的采样无法按 `#### 数字` 格式收尾，输出稳定性较差；Flash 的格式稳定性明显更好。

### 关键对比（提示工程的增量贡献）

**Qwen2.5-3B**
- E2 − E1 = **+46.7pp**：CoT 是最大单点增益
- E3 − E2 = **0**：Few-shot 在 CoT 之上无额外收益
- E4 − E2 = **+16.7pp**：自洽性有效
- E5 − E3 = +6.7pp，但 E5 − E4 = **−10pp**：少样本 + 自洽性组合在个别题上反而略降

**DeepSeek-V4-Flash**
- E2 − E1 = 0，E3 − E2 = 0，E4 − E2 = +3.4pp，E5 − E4 = 0
- 各项接近上限，提示工程只剩边际作用

### 主要结论

1. **模型能力差距远大于提示工程差距**：两模型 E1 基线差约 73pp，但通过提示工程（CoT、自洽性）可以缩小一定差距。
2. **CoT 对轻量模型是最大增益来源**（Qwen +46.7pp）；对推理类模型（Flash）几乎无效，因为它在 E1 时也会先在内部推理（输出带 `reasoning_content`）。
3. **自洽性在 Qwen 上增益明显**（+16.7pp），在接近天花板的 Flash 上只剩 +3.4pp。
4. **E5 vs E4 的 −10pp 是投票噪声而非真实能力倒退**：4 题失分 + 1 题反超（净 −3 题）。失分主因两类：
   - **错误共识**（q3、q15）：Few-shot 例题把 5 次采样中的多数收敛到同一条错误路径，投票"自信地错"；**（在此之前，从没有想过Few-shot的机制会使错误答案也收敛）**
   - **平局翻车**（q2、q26）：正确/错误各 2 票打平，按出现顺序取先出现的错误答案（q20 反超也是同一机制碰巧取对）。
   - 这些题在 E4 中胜者票数也只有 2–4/5，属于模型本就不确定的题目。
5. **不要忽略评估和忽视 Token 成本**：没有测试的 Prompt 优化是盲目的。Self-Consistency 和 ToT 的成本可能很高，需要权衡。
6. **不要过度追求复杂技术**：比如在DeepSeek-V4-Flash上，简单的 Zero-Shot CoT 往往就能解决 90% 的问题

### 局限与注意事项

- 样本量 30，单策略 95% 置信区间约 ±8pp，E2–E5 之间的细微差异需谨慎解读。
- Flash 的 E1 基线高并非"指令遵循好"，而是其原生推理能力所致（reasoning 模型）。
- 配置差异：Qwen 用 `max_tokens=512`（默认），Flash 用 1024（推理模型防截断）。
- `sample_answers` 中的 `null` 表示该次采样未能解析出 `####` 答案。

### 结果文件

- results.json：本地 Qwen2.5-3B 每题全部策略的答案与原始采样文本
- results_deepseek.json：DeepSeek-V4-Flash 对应结果