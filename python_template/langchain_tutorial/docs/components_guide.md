# LangChain 组件使用指南

LangChain 提供了丰富的组件，使开发者能够构建强大的语言模型应用。本指南将详细介绍 LangChain 的各个核心组件，并提供具体使用方法和示例。

## 1. 语言模型 (LLMs & Chat Models)

语言模型是 LangChain 的核心组件，主要分为两类：

### 1.1 LLM (Language Model)

用于生成文本的语言模型，接收文本输入并生成文本输出。

**主要特点：**
- 接收原始字符串作为输入
- 返回字符串作为输出
- 通常使用 `predict()` 方法调用

**常用模型：**
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- Cohere
- Hugging Face 开源模型
- 本地运行的开源模型

详细使用方法请参考 `code_examples/component_examples/llm_examples.py` 文件。

### 1.2 Chat Models

专为对话设计的语言模型，处理消息列表作为输入，生成消息作为输出。

**主要特点：**
- 接收聊天消息列表（通常包含系统、用户和助手消息）
- 返回一个消息对象
- 通常使用 `predict_messages()` 方法调用

**常用模型：**
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- Llama 2
- Mistral
- Gemini

详细使用方法请参考 `code_examples/component_examples/llm_examples.py` 文件。

## 2. 提示词 (Prompts)

提示词组件是用于管理、构建和优化发送给语言模型的提示的一系列工具。

### 2.1 提示模板 (Prompt Templates)

提示模板允许你创建可重用的提示结构，通过填充变量来生成最终提示。

**主要功能：**
- 参数化提示
- 格式化和验证输入
- 处理多种输入类型

### 2.2 聊天提示模板 (Chat Prompt Templates)

专为聊天模型设计的提示模板，用于构建消息列表。

**主要功能：**
- 创建消息序列
- 组合多个消息模板
- 定义角色和内容

### 2.3 示例选择器 (Example Selectors)

用于在少样本学习 (few-shot learning) 场景中选择最相关的示例。

**主要类型：**
- 基于相似度的选择器
- 基于长度的选择器
- 随机选择器

详细使用方法请参考 `code_examples/component_examples/prompt_examples.py` 文件。

## 3. 记忆 (Memory)

记忆组件用于存储和检索与语言模型交互的历史信息，使模型能够维持上下文。

### 3.1 对话记忆类型

**常用记忆类型：**
- 对话缓冲记忆 (ConversationBufferMemory)
- 对话缓冲窗口记忆 (ConversationBufferWindowMemory)
- 对话实体记忆 (ConversationEntityMemory)
- 对话摘要记忆 (ConversationSummaryMemory)
- 对话知识图谱记忆 (ConversationKGMemory)

### 3.2 记忆的集成方式

- 在链中使用记忆
- 在智能体中使用记忆
- 自定义记忆实现

详细使用方法请参考 `code_examples/component_examples/memory_examples.py` 文件。

## 4. 链 (Chains)

链是将多个组件组合成一个连贯操作序列的方式，使你能够构建复杂的语言模型应用。

### 4.1 常用链类型

- LLM 链 (LLMChain) - 将提示模板与语言模型连接
- 顺序链 (SequentialChain) - 按顺序执行多个链
- 路由链 (RouterChain) - 根据输入动态选择执行路径
- 转换链 (TransformChain) - 转换数据而不调用语言模型
- 检索链 (RetrievalChain) - 结合检索器和语言模型
- 问答链 (QAChain) - 专门用于问答应用的链

### 4.2 链的组合方式

- 顺序组合
- 平行组合
- 条件执行

详细使用方法请参考 `code_examples/component_examples/chain_examples.py` 文件。

## 5. 文档处理 (Document Loaders & Text Splitters)

文档处理组件用于加载、处理和拆分文档，为基于文档的应用奠定基础。

### 5.1 文档加载器 (Document Loaders)

用于从各种来源加载文档的组件。

**支持的文档类型：**
- 文本文件
- PDF
- HTML/网页
- Markdown
- CSV/Excel
- JSON/YAML
- 电子邮件
- 聊天记录
- 代码文件
- 数据库

### 5.2 文本分割器 (Text Splitters)

用于将长文档分割成较小的块，以便语言模型处理。

**常用分割器：**
- 字符文本分割器 (CharacterTextSplitter)
- 令牌文本分割器 (TokenTextSplitter)
- 递归字符文本分割器 (RecursiveCharacterTextSplitter)
- Markdown 文本分割器 (MarkdownTextSplitter)
- 代码文本分割器 (CodeTextSplitter)

详细使用方法请参考 `code_examples/component_examples/document_examples.py` 文件。

## 6. 检索器 (Retrievers)

检索器是用于从向量存储或其他数据源检索相关信息的组件。

### 6.1 检索器类型

- 向量存储检索器 (VectorStoreRetriever)
- 多查询检索器 (MultiQueryRetriever)
- 时间加权检索器 (TimeWeightedRetriever)
- 父文档检索器 (ParentDocumentRetriever)
- 自查询检索器 (SelfQueryRetriever)
- 上下文压缩检索器 (ContextualCompressionRetriever)

### 6.2 检索增强生成 (RAG)

结合检索器和语言模型的技术，使模型能够基于检索到的信息生成响应。

详细使用方法请参考 `code_examples/component_examples/document_examples.py` 文件。

## 7. 工具 (Tools)

工具是执行特定任务的函数或接口，可以由智能体调用来与外部系统交互。

### 7.1 内置工具

- 搜索工具 (SearchTools)
- 计算工具 (MathTools)
- Shell 工具 (ShellTools)
- 检索工具 (RetrievalTools)
- API 工具 (APITools)
- 文件操作工具 (FileTools)

### 7.2 自定义工具

- 基于函数的工具
- 结构化工具
- Lambda 工具
- Pydantic 工具

详细使用方法请参考 `code_examples/component_examples/tools_examples.py` 文件。

## 8. 智能体 (Agents)

智能体是 LangChain 的高级概念，它结合语言模型和工具，能够自主规划和执行任务。

### 8.1 智能体类型

- ReAct 智能体
- OpenAI 函数智能体
- Plan-and-Execute 智能体
- MRKL 智能体
- ChatGPT 插件智能体
- 自解释智能体

### 8.2 智能体执行器 (Agent Executor)

智能体执行器负责运行智能体，处理工具调用和错误，并维护状态。

详细使用方法请参考 `code_examples/agent_examples` 目录下的示例文件。

## 9. 输出处理器 (Output Parsers)

输出处理器用于将语言模型的原始输出转换为结构化格式。

### 9.1 常用输出处理器

- 结构化输出解析器 (StructuredOutputParser)
- Pydantic 输出解析器 (PydanticOutputParser)
- 枚举输出解析器 (EnumOutputParser)
- 列表输出解析器 (ListOutputParser)
- 日期时间输出解析器 (DatetimeOutputParser)

### 9.2 输出格式

- JSON
- XML
- Markdown
- YAML
- 自定义格式

详细使用方法可以在各个示例文件中找到相关用法。

## 10. 评估 (Evaluation)

LangChain 提供了一系列工具来评估语言模型应用的性能。

### 10.1 评估方法

- 字符串评估器 (StringEvaluator)
- 轨迹评估器 (TrajectoryEvaluator)
- 对比评估 (Comparative Evaluation)
- 自动评估 (Auto-Evaluation)

### 10.2 常用评估指标

- 正确性 (Correctness)
- 相关性 (Relevance)
- 一致性 (Coherence)
- 有害性 (Harmfulness)
- 偏见性 (Bias)

## 11. 跟踪和调试 (Tracing & Debugging)

LangChain 提供了跟踪和调试工具，帮助开发者理解和优化他们的应用。

### 11.1 LangSmith

LangChain 的官方跟踪和评估平台。

### 11.2 回调 (Callbacks)

自定义回调函数来监控和修改 LangChain 组件的行为。

## 12. 数据连接 (Data Connection)

帮助语言模型有效地连接和利用数据的组件。

### 12.1 文档加载

从各种来源加载文档。

### 12.2 文档转换

处理和转换文档。

### 12.3 文本嵌入

将文本转换为向量表示。

### 12.4 向量存储

存储和检索向量嵌入。

## 小结

LangChain 组件提供了构建语言模型应用的全套工具。通过组合这些组件，你可以创建强大的应用程序，从简单的聊天机器人到复杂的智能体系统。

在后续章节中，我们将通过具体的代码示例，深入探讨如何使用这些组件构建实际应用。