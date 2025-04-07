# LangChain 框架中文教程

这是一个全面的 LangChain 框架中文教程，基于官方文档和 GitHub 仓库，旨在帮助中文开发者快速上手 LangChain 框架。

## 项目概述

LangChain 是一个用于开发由语言模型驱动的应用程序的框架。它的目标是为创建基于大型语言模型（LLMs）的应用程序提供工具和抽象，使得开发者能够更轻松地构建复杂的语言模型应用。

本教程使用 Python 3.10 实现所有代码示例，并提供详细的中文说明和注释。

## 目录结构

```
langchain_tutorial/
├── docs/                           # 框架概念和使用文档
│   ├── langchain_overview.md      # LangChain 框架概述
│   ├── components_guide.md        # 组件使用指南
│   ├── vector_stores_guide.md     # 向量存储指南
│   ├── memory_guide.md            # 记忆组件指南
│   ├── agent_guide.md             # 智能体使用指南
│   └── workflow_guide.md          # 工作流编排指南
│
├── code_examples/                  # 代码示例
│   ├── component_examples/        # 各组件使用案例
│   │   ├── llm_examples.py        # LLM 组件示例
│   │   ├── prompt_examples.py     # 提示词组件示例
│   │   ├── memory_examples.py     # 记忆组件示例
│   │   ├── chain_examples.py      # 链组件示例
│   │   ├── vector_store_examples.py # 向量存储示例
│   │   ├── document_examples.py   # 文档处理组件示例
│   │   └── tools_examples.py      # 工具组件示例
│   │
│   ├── agent_examples/            # 智能体示例
│   │   ├── single_agent_example.py # 单智能体案例
│   │   └── multi_agent_example.py  # 多智能体案例
│   │
│   ├── search_agent_integration/  # 搜索集成示例
│   │   ├── basic_search_agent.py  # 基础搜索智能体
│   │   └── advanced_search_agent.py # 高级搜索智能体
│   │
│   └── workflow_examples/         # 工作流示例
│       └── complex_workflow.py    # 综合多智能体工作流案例
│
└── utils/                         # 工具函数和辅助模块
    ├── api_keys.py               # API 密钥管理
    └── helper_functions.py       # 辅助函数
```

## 使用方法

1. 确保你已安装 Python 3.10 或更高版本
2. 安装必要的依赖：`pip install -r requirements.txt`
3. 按照文档指南逐步学习 LangChain 框架的各个方面
4. 运行代码示例以深入理解各个组件和功能

## 主要内容

1. **组件使用指南**：详细介绍 LangChain 的各个核心组件，并提供实际调用案例
   - LLM 和聊天模型的调用和配置
   - 提示词管理和工程化
   - 记忆组件的使用和定制
   - 链的构建和组合
   - 向量存储和检索增强生成(RAG)
   - 文档加载和处理
   - 工具集成和调用
2. **智能体编排指南**：讲解单智能体和多智能体的设计与实现方法
3. **搜索与智能体集成**：展示如何将搜索功能与智能体结合
4. **多智能体工作流**：提供一个综合性的多智能体编排工作流案例

## 注意事项

- 所有示例代码需要配置相应的 API 密钥才能运行
- 请在 `utils/api_keys.py` 文件中设置你的 API 密钥
- 示例代码主要使用 OpenAI 的模型，但也提供了使用其他模型的替代方案
