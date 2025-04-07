# LangChain 搜索与智能体集成示例

本目录包含展示LangChain框架中搜索功能与智能体集成的各种实现方式的代码示例。

## 文件说明

1. `basic_search_agent.py` - 基础搜索智能体示例
   - 演示三种基本的搜索-智能体集成模式：
     - 直接检索QA链
     - 检索作为智能体工具
     - 上下文增强智能体

2. `advanced_search_agent.py` - 高级搜索智能体示例
   - 演示更复杂的搜索增强技术：
     - 查询转换/重写
     - 上下文压缩
     - 多轮对话记忆

## 集成方法概述

1. **直接检索模式**
   - 使用检索QA链直接查询向量存储
   - 最简单的实现方式
   - 适用于单轮、直接的问答场景

2. **检索工具模式**
   - 将检索器包装为智能体工具
   - 让智能体决定何时搜索信息
   - 适用于复杂任务和开放域问答

3. **上下文增强模式**
   - 在处理前检索相关信息
   - 作为上下文提供给智能体
   - 适用于知识密集型任务

## 使用说明

1. 确保安装所需依赖：
   ```bash
   pip install langchain langchain_community faiss-cpu openai
   ```

2. 设置OpenAI API密钥：
   ```python
   import os
   os.environ["OPENAI_API_KEY"] = "your-api-key"
   ```

3. 运行示例：
   ```bash
   python basic_search_agent.py
   python advanced_search_agent.py
   ```