# LangChain 框架学习资源索引

## 项目概述

本项目收集并整理了 LangChain 框架的相关资源，包括框架组件说明、智能体实现方法、搜索集成和多智能体工作流的设计与实现。所有资料基于 LangChain 官方文档（[https://python.langchain.com/docs/introduction/](https://python.langchain.com/docs/introduction/)）整理而成。

## 综合指南

- [LangChain 框架完整指南](langchain_complete_guide.md) - 全面介绍 LangChain 的架构、组件和使用方法

## 文档结构

### 基础文档

- [文档结构](../docs/document_structure.md) - LangChain 文档的整体结构
- [框架组件](../docs/components.md) - LangChain 的核心组件概述
- [智能体](../docs/agents.md) - 智能体系统概览
- [框架总览](../docs/overview.md) - LangChain 框架的总体架构

### LangChain 表达式语言 (LCEL)

- [LCEL 介绍](../docs/lcel_intro.md) - LangChain 表达式语言基础
- [LCEL 接口](../docs/lcel_interface.md) - LCEL 接口说明
- [LCEL 食谱](../docs/lcel_cookbook.md) - LCEL 使用示例

### 组件详解

- [模型输入输出](../docs/components/model_io.md) - 语言模型接口
- [记忆系统](../docs/components/memory.md) - 对话历史管理
- [提示模板](../docs/components/prompt_templates.md) - 提示工程
- [输出解析器](../docs/components/output_parsers.md) - 结构化输出处理
- [链](../docs/components/chains.md) - 组件组合
- [检索系统](../docs/components/retrieval.md) - 数据连接和检索

### 单智能体实现

- [智能体执行器](../docs/single_agent/agent_executor.md) - 智能体执行管理
- [ReAct 智能体](../docs/single_agent/react_agent.md) - 结合推理和行动的智能体
- [OpenAI 函数智能体](../docs/single_agent/openai_functions.md) - 基于函数调用的智能体
- [结构化聊天智能体](../docs/single_agent/structured_chat.md) - 结构化对话智能体
- [计划执行智能体](../docs/single_agent/plan_execute.md) - 规划与执行智能体

### 多智能体编排

- [智能体监督](../docs/multi_agent/agent_supervision.md) - 多智能体监督框架
- [团队监督者模式](../docs/multi_agent/team_supervisor.md) - 团队协作模式
- [经理-工人模式](../docs/multi_agent/manager_worker.md) - 层级协作模式
- [计划执行模式](../docs/multi_agent/plan_executor.md) - 规划与执行分离模式
- [CrewAI 集成](../docs/multi_agent/crewai_integration.md) - CrewAI 多智能体框架集成

### 搜索与智能体集成

- [搜索与智能体集成总结](../docs/search_agent_integration/search_agent_summary.md) - 综合概述
- [搜索工具](../docs/search_agent_integration/search_tools.md) - 智能体可用的搜索工具
- [搜索工具包](../docs/search_agent_integration/search_toolkit.md) - 搜索工具集成包
- [向量存储](../docs/search_agent_integration/vector_stores.md) - 向量数据库集成
- [检索器](../docs/search_agent_integration/retrievers.md) - 检索增强系统
- [向量数据库聊天](../docs/search_agent_integration/agent_with_retrieval_tool.md) - 集成检索工具的聊天智能体

### 智能体编排指南

- [智能体编排模式](../docs/agent_patterns_summary.md) - 单智能体和多智能体编排模式总结
- [智能体编排指南](../docs/agent_orchestration_guide.md) - 智能体协作系统设计指南

## 代码示例

### 智能体示例

- [单智能体示例](../code_examples/agent_examples/single_agent_example.py) - 不同类型单智能体实现
- [多智能体示例](../code_examples/agent_examples/multi_agent_example.py) - 多智能体协作模式实现
- [RAG 智能体示例](../code_examples/agent_examples/rag_agent_example.py) - 检索增强智能体实现
- [README](../code_examples/agent_examples/README.md) - 示例代码使用说明

### 搜索智能体集成示例

- [基础搜索智能体](../code_examples/search_agent_integration/basic_search_agent.py) - 基本搜索集成模式
- [高级搜索智能体](../code_examples/search_agent_integration/advanced_search_agent.py) - 高级搜索技术
- [README](../code_examples/search_agent_integration/README.md) - 搜索集成示例使用说明

### 多智能体工作流示例

- [多智能体工作流](../code_examples/multi_agent_demo/multi_agent_workflow.py) - 完整多智能体系统示例
- [专家智能体](../code_examples/multi_agent_demo/agents/specialist_agents.py) - 专业领域智能体实现
- [自定义工具](../code_examples/multi_agent_demo/tools/custom_tools.py) - 智能体工具扩展
- [LCEL 示例](../code_examples/multi_agent_demo/lcel_example.py) - LangChain 表达式语言示例
- [README](../code_examples/multi_agent_demo/README.md) - 多智能体演示使用说明

## 资源来源

所有资料基于 LangChain 官方文档：[https://python.langchain.com/docs/introduction/](https://python.langchain.com/docs/introduction/)
