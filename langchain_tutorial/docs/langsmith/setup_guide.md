# LangSmith 设置指南

## 什么是 LangSmith？

LangSmith 是 LangChain 团队开发的一个统一开发者平台，用于构建、测试、评估和监控基于大型语言模型（LLM）的应用程序。它提供了一套工具，帮助开发者更好地理解、调试和改进他们的 LLM 应用。

## 注册 LangSmith

在开始使用 LangSmith 前，您需要创建一个账户：

1. 访问 [LangSmith 网站](https://smith.langchain.com/)
2. 点击 "Sign Up" 创建新账户或使用现有账户登录
3. 完成注册流程，您将获得访问 LangSmith 控制台的权限

## API 密钥设置

要将您的应用与 LangSmith 集成，您需要获取 API 密钥：

1. 登录到 LangSmith 控制台
2. 点击右上角的个人资料图标，然后选择 "API 密钥"
3. 创建新的 API 密钥并复制它

将此 API 密钥设置为环境变量：

```bash
# 在 Linux/MacOS 上
export LANGCHAIN_API_KEY="your-api-key-here"
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_PROJECT="my-project-name"  # 可选，指定项目名称

# 在 Windows 上
set LANGCHAIN_API_KEY=your-api-key-here
set LANGCHAIN_TRACING_V2=true
set LANGCHAIN_PROJECT=my-project-name
```

您也可以在代码中直接设置这些变量：

```python
import os

os.environ["LANGCHAIN_API_KEY"] = "your-api-key-here"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "my-project-name"
```

## 安装 LangSmith 客户端库

LangSmith 客户端库可以通过 pip 安装：

```bash
pip install langsmith
```

如果您已经安装了 LangChain，那么从 LangChain 版本 0.0.267 开始，LangSmith 跟踪功能已经被自动包含在其中。

## 项目设置

### 创建项目

LangSmith 中的项目用于组织和管理相关的运行和数据集：

1. 在 LangSmith 控制台中，点击 "Projects" 标签
2. 点击 "Create New Project" 按钮
3. 输入项目名称和可选的描述
4. 点击 "Create"

### 指定项目

您可以通过以下方式在代码中指定要使用的项目：

```python
# 方法 1：环境变量
os.environ["LANGCHAIN_PROJECT"] = "my-project-name"

# 方法 2：使用 langchain.smith 上下文管理器
from langchain.smith import RunEvalConfig

with RunEvalConfig(project_name="my-project-name"):
    # 您的 LangChain 代码在这里
    chain.invoke({"input": "Hello world"})
```

## 为监控配置应用

### 基本跟踪

要启用基本的 LangSmith 跟踪，只需设置环境变量：

```python
os.environ["LANGCHAIN_TRACING_V2"] = "true"
```

这将自动捕获所有 LangChain 运行并将其发送到 LangSmith。

### 自定义跟踪

您可以添加额外的元数据和标签来丰富您的跟踪数据：

```python
from langchain.callbacks.tracers import LangSmithTracer

# 创建一个带有自定义元数据的跟踪器
tracer = LangSmithTracer(
    project_name="my-custom-project",
    tags=["production", "user-query"],
    metadata={"user_id": "user-123", "session_id": "abc-xyz"}
)

# 将跟踪器添加到链的调用中
result = chain.invoke({"input": "Hello"}, config={"callbacks": [tracer]})
```

### 记录自定义步骤

您可以使用 LangSmith API 记录不是 LangChain 组件的自定义步骤：

```python
from langsmith import Client

client = Client()

def process_data(data):
    # 创建一个新的运行
    with client.trace("data_processing", name="Process Raw Data") as run:
        try:
            # 您的数据处理代码
            result = transform_data(data)
            # 记录输出
            run.end(outputs={"processed_data": result})
            return result
        except Exception as e:
            # 记录错误
            run.end(error=str(e))
            raise
```

## LangChain 组件的自动监控

LangChain 与 LangSmith 深度集成，大多数 LangChain 组件都会自动发送遥测数据到 LangSmith，包括：

- LLM 和 ChatModel 调用
- 提示模板渲染
- 链和序列执行
- 检索器查询
- 工具和代理操作

您只需要设置环境变量，无需额外的代码修改。

## 跨环境跟踪

### 开发环境

在开发环境中，您可能希望启用详细跟踪以便调试：

```python
# 开发环境配置
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "my-app-dev"
```

### 生产环境

在生产环境中，您可能希望限制跟踪范围或添加额外标签：

```python
# 生产环境配置
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "my-app-prod"

# 使用标签区分不同环境或功能
from langchain.callbacks.tracers import LangSmithTracer

tracer = LangSmithTracer(
    project_name="my-app-prod",
    tags=["production", "region-asia"],
    metadata={"deployment_id": "v2.5"}
)
```

## 客户端库 API

### 手动创建运行

您可以使用 LangSmith 客户端库直接创建和管理运行：

```python
from langsmith import Client

client = Client()

# 创建新的运行
run = client.create_run(
    name="Custom Process",
    run_type="chain",
    inputs={"query": "What's the weather like?"},
    project_name="my-project"
)

try:
    # 执行一些操作
    output = process_query("What's the weather like?")
    
    # 更新运行状态为成功
    client.update_run(
        run.id,
        outputs={"result": output},
        end_time=datetime.datetime.utcnow()
    )
except Exception as e:
    # 记录错误
    client.update_run(
        run.id,
        error=str(e),
        end_time=datetime.datetime.utcnow()
    )
```

### 查询运行

您可以编程方式查询和分析 LangSmith 中的运行：

```python
from langsmith import Client

client = Client()

# 获取特定项目中的运行
runs = client.list_runs(
    project_name="my-project",
    filter={
        "start_time": {"$gt": "2023-07-01"},
        "error": {"$exists": False}
    }
)

# 分析运行结果
for run in runs:
    print(f"Run ID: {run.id}, Latency: {run.latency_ms} ms")
```

## 故障排除

### 常见问题

1. **没有看到任何运行数据**
   - 确保 `LANGCHAIN_API_KEY` 和 `LANGCHAIN_TRACING_V2` 环境变量已正确设置
   - 检查网络连接，确保您的环境可以访问 LangSmith API

2. **运行数据不完整**
   - 确保运行已正确结束，所有链和代理都已完成执行
   - 检查是否有未捕获的异常中断了跟踪过程

3. **API 密钥错误**
   - 验证您的 API 密钥是否正确
   - 检查 API 密钥是否已过期或被撤销

### 调试提示

启用详细日志记录以帮助调试 LangSmith 集成：

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("langsmith").setLevel(logging.DEBUG)
```

## 其他资源

- [LangSmith 官方文档](https://docs.smith.langchain.com/)
- [LangSmith API 参考](https://docs.smith.langchain.com/api-reference/)
- [LangChain 集成指南](https://python.langchain.com/docs/langsmith/)