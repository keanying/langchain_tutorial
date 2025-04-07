# LangSmith 监控指南

## 概述

LangSmith 提供了强大的监控和可观察性功能，帮助开发者追踪、分析和优化他们的 LLM 应用程序。本指南将介绍如何使用 LangSmith 对您的应用进行全面监控。

## 控制台监控

### 仪表盘概览

LangSmith 控制台提供了一个全面的仪表盘，展示您应用的关键指标：

- **运行总数**：追踪应用的使用情况
- **平均延迟**：监控响应时间
- **错误率**：识别潜在问题
- **令牌使用情况**：追踪成本
- **每日活跃用户**：了解用户参与度

### 实时监控

控制台的「Live」标签页允许您实时查看应用中正在发生的运行：

1. 登录到 LangSmith 控制台
2. 导航到您的项目
3. 点击「Live」标签查看实时运行数据
4. 使用过滤器选项关注特定类型的运行

## 追踪详细运行

### 运行检查器

运行检查器是调试和分析单个运行的强大工具：

1. 从运行列表中点击任何运行
2. 查看完整的执行跟踪，包括：
   - 输入和输出
   - 中间步骤
   - 运行时间线
   - 令牌使用情况
   - 错误和异常

### 追踪复杂工作流

对于复杂的工作流（如智能体和链的组合），LangSmith 提供了详细的执行图：

- 树形视图显示嵌套组件的层次结构
- 时间轴视图显示组件执行的时间顺序
- 节点间的边表示数据流

## 性能分析

### 延迟分析

监控和分析应用的延迟对用户体验至关重要：

```python
from langsmith import Client

client = Client()
runs = client.list_runs(
    project_name="my-production-app",
    start_time="-7d"  # 过去7天
)

# 计算平均延迟
latencies = [run.latency_ms for run in runs if run.latency_ms is not None]
avg_latency = sum(latencies) / len(latencies) if latencies else 0
print(f"平均延迟: {avg_latency} 毫秒")

# 分析延迟分布
import matplotlib.pyplot as plt
plt.hist(latencies, bins=50)
plt.title("运行延迟分布")
plt.xlabel("延迟 (毫秒)")
plt.ylabel("频率")
plt.savefig("latency_distribution.png")
```

### 令牌使用分析

监控令牌使用对控制成本和预算至关重要：

```python
from langsmith import Client

client = Client()
runs = client.list_runs(
    project_name="my-production-app",
    run_type="llm",  # 仅LLM调用
    start_time="-30d"  # 过去30天
)

# 按日期汇总令牌使用
from collections import defaultdict
import datetime

daily_tokens = defaultdict(lambda: {"prompt": 0, "completion": 0})

for run in runs:
    if not run.metrics:
        continue
        
    date_str = run.start_time.strftime("%Y-%m-%d")
    prompt_tokens = run.metrics.get("prompt_tokens", 0)
    completion_tokens = run.metrics.get("completion_tokens", 0)
    
    daily_tokens[date_str]["prompt"] += prompt_tokens
    daily_tokens[date_str]["completion"] += completion_tokens

# 打印每日令牌使用
for date, tokens in sorted(daily_tokens.items()):
    print(f"{date}: 提示令牌 {tokens['prompt']}, 完成令牌 {tokens['completion']}")
```

## 错误监控

### 跟踪和分析错误

识别和分析应用中的错误模式：

```python
from langsmith import Client
from collections import Counter

client = Client()
error_runs = client.list_runs(
    project_name="my-production-app",
    filter={"error": {"$exists": True}},
    start_time="-14d"  # 过去14天
)

# 计算错误类型分布
error_types = Counter()
for run in error_runs:
    # 提取错误类型（通常是异常类名）
    error_message = run.error
    if error_message:
        error_type = error_message.split(":")[0] if ":" in error_message else error_message
        error_types[error_type] += 1

# 打印最常见的错误类型
for error_type, count in error_types.most_common(10):
    print(f"{error_type}: {count} 次发生")
```

### 设置错误警报

配置自动化警报以在出现问题时通知您：

1. 在 LangSmith 控制台中，导航到 "Monitoring" > "Alerts"
2. 点击 "Create Alert"
3. 配置警报条件，例如：
   - 错误率超过 5%
   - 平均延迟超过 3000 毫秒
   - 特定运行类型的失败
4. 设置通知渠道（电子邮件、Slack 等）
5. 设置警报频率和静默期

## 用户反馈和评估

### 收集用户反馈

LangSmith 允许您收集和分析用户对 LLM 响应的反馈：

```python
from langsmith import Client

client = Client()

# 记录用户对特定运行的反馈
client.create_feedback(
    run_id="run-id-here",
    key="thumbs",
    value="up",  # 或 "down"
    comment="用户觉得这个回复很有帮助",
)

# 查询带有用户反馈的运行
feedback_runs = client.list_runs(
    project_name="my-production-app",
    filter={"feedback": {"$exists": True}},
)

# 分析反馈分布
from collections import Counter

feedback_counts = Counter()
for run in feedback_runs:
    for feedback in client.list_feedback(run_id=run.id):
        if feedback.key == "thumbs":
            feedback_counts[feedback.value] += 1

print(f"点赞: {feedback_counts['up']}, 踩: {feedback_counts['down']}")
```

### 构建评估循环

使用 LangSmith 创建评估循环，自动评估模型输出质量：

```python
from langsmith import Client
from langsmith.evaluation import RunEvaluator

# 创建评估器
class RelevanceEvaluator(RunEvaluator):
    def evaluate_run(self, run):
        # 提取查询和响应
        query = run.inputs.get("query", "")
        response = run.outputs.get("result", "")
        
        # 使用 LLM 评估响应的相关性
        evaluation_result = self.evaluate_relevance(query, response)
        
        # 返回评估结果
        return {
            "relevance_score": evaluation_result["score"],
            "reasoning": evaluation_result["reasoning"]
        }
        
    def evaluate_relevance(self, query, response):
        # 实现使用 LLM 评估相关性的逻辑
        # 在实际应用中，这将调用 LLM API
        pass

# 应用评估器到项目
client = Client()
client.run_evaluator(
    evaluator=RelevanceEvaluator(),
    project_name="my-production-app",
    filter={"run_type": "chain"},
    concurrency=5  # 并行评估运行
)
```

## 深度分析

### 追踪关键指标

定期追踪和分析关键性能指标：

```python
from langsmith import Client
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

client = Client()

# 设置时间范围
end_date = datetime.utcnow()
start_date = end_date - timedelta(days=30)

# 获取运行数据
runs = client.list_runs(
    project_name="my-production-app",
    start_time=start_date.isoformat(),
    end_time=end_date.isoformat()
)

# 转换为 DataFrame 进行分析
run_data = []
for run in runs:
    run_data.append({
        "id": run.id,
        "start_time": run.start_time,
        "latency_ms": run.latency_ms,
        "run_type": run.run_type,
        "error": run.error is not None,
        "prompt_tokens": run.metrics.get("prompt_tokens", 0) if run.metrics else 0,
        "completion_tokens": run.metrics.get("completion_tokens", 0) if run.metrics else 0,
        "total_tokens": (run.metrics.get("prompt_tokens", 0) + run.metrics.get("completion_tokens", 0)) if run.metrics else 0
    })

df = pd.DataFrame(run_data)

# 设置日期索引
df["date"] = pd.to_datetime(df["start_time"]).dt.date
df.set_index("date", inplace=True)

# 分析每日指标
daily_metrics = df.groupby(df.index).agg({
    "id": "count",  # 运行总数
    "latency_ms": "mean",  # 平均延迟
    "error": "mean",  # 错误率
    "total_tokens": "sum"  # 总令牌使用
})

daily_metrics.columns = ["运行总数", "平均延迟(ms)", "错误率", "令牌使用量"]

# 可视化趋势
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("LLM 应用性能指标趋势")

daily_metrics["运行总数"].plot(ax=axes[0, 0], title="每日运行总数")
daily_metrics["平均延迟(ms)"].plot(ax=axes[0, 1], title="平均延迟趋势")
daily_metrics["错误率"].plot(ax=axes[1, 0], title="错误率趋势")
daily_metrics["令牌使用量"].plot(ax=axes[1, 1], title="每日令牌使用量")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("performance_trends.png")
```

### 比较模型性能

使用 LangSmith 比较不同模型或提示的性能：

```python
from langsmith import Client
import pandas as pd

client = Client()

# 获取使用不同模型的运行
model_a_runs = client.list_runs(
    project_name="model-comparison",
    filter={"metadata.model": "gpt-4"}
)

model_b_runs = client.list_runs(
    project_name="model-comparison",
    filter={"metadata.model": "claude-2"}
)

# 提取关键指标
def extract_metrics(runs):
    metrics = []
    for run in runs:
        if run.outputs and run.latency_ms:
            metrics.append({
                "latency_ms": run.latency_ms,
                "tokens": run.metrics.get("total_tokens", 0) if run.metrics else 0,
                "error": run.error is not None
            })
    return pd.DataFrame(metrics)

model_a_df = extract_metrics(model_a_runs)
model_b_df = extract_metrics(model_b_runs)

# 比较性能
print("模型 A (GPT-4) 性能:")
print(f"平均延迟: {model_a_df['latency_ms'].mean():.2f} ms")
print(f"平均令牌使用: {model_a_df['tokens'].mean():.2f}")
print(f"错误率: {model_a_df['error'].mean() * 100:.2f}%")

print("\n模型 B (Claude-2) 性能:")
print(f"平均延迟: {model_b_df['latency_ms'].mean():.2f} ms")
print(f"平均令牌使用: {model_b_df['tokens'].mean():.2f}")
print(f"错误率: {model_b_df['error'].mean() * 100:.2f}%")
```

## 持续改进

### 建立改进循环

使用 LangSmith 数据创建持续改进循环：

1. **收集数据**：捕获运行、反馈和评估
2. **分析性能**：识别模式和问题区域
3. **进行实验**：测试改进的提示或模型
4. **比较结果**：评估变更的影响
5. **部署更新**：将改进部署到生产环境

### 创建测试数据集

从生产数据创建测试数据集，用于未来的评估：

```python
from langsmith import Client

client = Client()

# 获取具有代表性的运行
representative_runs = client.list_runs(
    project_name="my-production-app",
    filter={
        "run_type": "chain",
        "error": {"$exists": False}  # 仅成功的运行
    },
    limit=100  # 选择一个合理的样本量
)

# 创建新数据集
dataset = client.create_dataset(
    "production-examples",
    description="从生产中提取的代表性查询示例"
)

# 向数据集添加示例
for run in representative_runs:
    client.create_example(
        inputs=run.inputs,
        outputs=run.outputs,
        dataset_id=dataset.id
    )

print(f"创建了包含 {len(representative_runs)} 个示例的测试数据集")
```

## 最佳实践

### 结构化元数据

使用结构化元数据丰富您的监控数据：

```python
from langchain.callbacks.tracers import LangSmithTracer

# 创建带有丰富元数据的跟踪器
tracer = LangSmithTracer(
    project_name="customer-support-bot",
    metadata={
        "user_id": "user-123",
        "session_id": "session-456",
        "channel": "web",
        "locale": "zh-CN",
        "version": "1.2.3",
        "feature_flags": {
            "use_rag": True,
            "advanced_reasoning": False
        }
    }
)

# 使用跟踪器调用链
response = chain.invoke(
    {"query": "如何重置我的密码？"}, 
    config={"callbacks": [tracer]}
)
```

### 系统健康监控

创建全面的系统健康仪表盘：

1. 在 LangSmith 控制台中，导航到 "Monitoring" 标签
2. 点击 "Create Dashboard"
3. 添加关键指标部件，如：
   - 运行量趋势
   - 错误率图表
   - 延迟分布
   - 令牌使用量
4. 设置自动刷新和提醒阈值

### 敏感数据处理

确保遵守数据隐私要求：

```python
from langchain.callbacks.tracers import LangSmithTracer
import re

# 创建一个敏感数据过滤函数
def filter_sensitive_data(data):
    if isinstance(data, str):
        # 过滤电子邮件地址
        data = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[EMAIL]', data)
        # 过滤电话号码
        data = re.sub(r'\b(?:\d{3}[-.]?){1,4}\d{4}\b', '[PHONE]', data)
        # 过滤信用卡号
        data = re.sub(r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b', '[CREDIT_CARD]', data)
    
    return data

# 应用过滤器到跟踪器
class PrivacyAwareTracer(LangSmithTracer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _preprocess_inputs(self, inputs):
        if isinstance(inputs, dict):
            return {k: filter_sensitive_data(v) for k, v in inputs.items()}
        return filter_sensitive_data(inputs)

# 使用隐私感知跟踪器
tracer = PrivacyAwareTracer(project_name="secure-app")
```

## 结论

LangSmith 提供了一套全面的工具，用于监控、分析和改进您的 LLM 应用程序。通过采用本指南中的实践，您可以：

- 全面了解应用性能
- 快速识别和解决问题
- 优化成本和用户体验
- 建立数据驱动的持续改进流程

随着 LLM 应用的复杂性增加，强大的监控和可观察性变得至关重要。LangSmith 作为专门为 LLM 应用设计的监控解决方案，提供了必要的工具来确保您的应用既可靠又高效。