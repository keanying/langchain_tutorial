# LangChain 智能体和多智能体编排示例

本目录包含 LangChain 框架中单智能体和多智能体编排的实用示例代码。

## 文件说明

1. `single_agent_example.py` - 演示不同类型的单智能体实现
   - ReAct 智能体：结合推理和行动的智能体
   - OpenAI函数调用智能体：基于函数调用能力的智能体
   - 自定义工具智能体：使用自定义工具的智能体

2. `multi_agent_example.py` - 演示多智能体编排模式
   - 团队监督者模式：一个监督者智能体协调多个专家智能体工作
   - 计划执行模式：一个智能体制定计划，另一个智能体执行计划

3. `rag_agent_example.py` - 演示RAG与智能体结合的实现
   - 直接RAG问答：使用检索增强生成直接回答问题
   - 智能体与RAG集成：智能体可以主动使用RAG工具获取知识

## 使用说明

1. 确保设置必要的环境变量：
   ```python
   os.environ["OPENAI_API_KEY"] = "your-api-key"
   ```

2. 安装所需依赖：
   ```bash
   pip install langchain langchain_community faiss-cpu openai
   ```

3. 运行示例：
   ```bash
   python single_agent_example.py
   python multi_agent_example.py
   python rag_agent_example.py
   ```

## 注意事项

- 这些示例旨在展示核心概念，实际应用时可能需要进一步配置和优化
- 代码中的某些功能可能需要特定的API访问权限
- RAG示例中使用了临时文件来模拟文档加载，实际应用中可以直接从真实数据源加载
