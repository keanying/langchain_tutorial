# LangChain 多智能体工作流示例

本示例展示了 LangChain 框架的核心组件和多智能体工作流程。示例包括：

1. 基本组件演示：提示模板、语言模型、链、记忆组件
2. RAG (检索增强生成) 示例：向量数据库、文档加载与分割、查询
3. 单智能体工作流：使用工具增强智能体能力
4. 多智能体协作系统：执行复杂任务的多智能体编排

## 环境设置

需要以下环境变量：
- OPENAI_API_KEY：OpenAI API 密钥

## 文件结构

- `multi_agent_workflow.py`：主要示例代码
- `data/`：示例数据文件
- `agents/`：定义的智能体
- `tools/`：可用工具

## 运行方式
python multi_agent_workflow.py
