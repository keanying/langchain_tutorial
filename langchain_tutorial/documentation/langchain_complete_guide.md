001|# LangChain 框架全面指南
002|
003|## 1. LangChain 框架概述
004|
005|LangChain 是一个用于开发由大型语言模型（LLMs）驱动的应用程序的框架，它提供了一系列组件和工具，使开发者能够创建复杂的、交互式的、基于语言模型的应用。
006|
007|### 1.1 框架核心理念
008|
009|LangChain 的设计理念围绕以下几个核心原则：
010|
011|- **组件化设计**：提供模块化的组件，可以独立使用或组合成复杂的系统
012|- **与语言模型的无缝集成**：优化与各种语言模型的交互方式
013|- **链式处理**：允许将多个组件组合成处理管道
014|- **状态管理**：提供记忆组件以维护对话历史和状态
015|- **工具集成**：允许语言模型与外部工具和系统交互
016|
017|### 1.2 LangChain 表达式语言 (LCEL)
018|
019|LangChain 表达式语言是一种声明式语言，用于组合 LangChain 的各种组件，具有以下特点：
020|
021|- 使用管道操作符 (`|`) 连接组件
022|- 支持同步和异步操作
023|- 内置错误处理和重试机制
024|- 支持流式传输和批处理
025|- 简化复杂链的构建过程
026|
027|示例：
028|```python
029|from langchain.chat_models import ChatOpenAI
030|from langchain.prompts import ChatPromptTemplate
031|from langchain.schema.output_parser import StrOutputParser
032|
033|prompt = ChatPromptTemplate.from_template("讲一个关于{topic}的笑话")
034|model = ChatOpenAI()
035|output_parser = StrOutputParser()
036|
037|chain = prompt | model | output_parser
038|
039|result = chain.invoke({"topic": "人工智能"})
040|```
041|
042|## 2. LangChain 核心组件
043|
044|LangChain 框架由多个核心组件构成，每个组件负责特定的功能：
045|
046|### 2.1 模型 (Models)
047|
048|模型组件是 LangChain 的核心，包括语言模型和嵌入模型：
049|
050|#### 2.1.1 语言模型 (LLMs/Chat Models)
051|
052|- **LLM**：文本输入，文本输出的模型（如 GPT-3.5, Llama 2）
053|- **ChatModel**：结构化输入（消息），结构化输出的模型（如ChatGPT, Claude）
054|
055|示例：
056|```python
057|from langchain.llms import OpenAI
058|from langchain.chat_models import ChatOpenAI
059|
060|# 传统LLM
061|llm = OpenAI(temperature=0)
062|result = llm.predict("一个 AI 走进了酒吧...")
063|
064|# 聊天模型
065|chat_model = ChatOpenAI(temperature=0)
066|messages = [
067|    SystemMessage(content="你是一个有幽默感的助手"),
068|    HumanMessage(content="讲一个 AI 笑话")
069|]
070|response = chat_model.predict_messages(messages)
071|```
072|
073|#### 2.1.2 嵌入模型 (Embeddings)
074|
075|嵌入模型将文本转换为数值向量，用于语义搜索和其他相似性比较：
076|
077|```python
078|from langchain.embeddings import OpenAIEmbeddings
079|
080|embeddings = OpenAIEmbeddings()
081|vector = embeddings.embed_query("Hello world")
082|```
083|
084|### 2.2 提示模板 (Prompts)
085|
086|提示模板用于构建结构化的提示：
087|
088|- **PromptTemplate**：构建简单的文本提示
089|- **ChatPromptTemplate**：构建聊天消息格式的提示
090|- **支持变量和条件逻辑**：动态构建提示
091|
092|示例：
093|```python
094|from langchain.prompts import PromptTemplate, ChatPromptTemplate
095|
096|# 基本提示模板
097|template = "给我提供关于{topic}的摘要，长度约{length}个字"
098|prompter = PromptTemplate.from_template(template)
099|prompt = prompter.format(topic="量子计算", length="100")
100|
101|# 聊天提示模板
102|template = "你是{role}，请回答关于{topic}的问题"
103|chat_prompter = ChatPromptTemplate.from_messages([
104|    ("system", template),
105|    ("human", "{question}")
106|])
107|```
108|
109|### 2.3 记忆 (Memory)
110|
111|记忆组件用于管理对话历史或持久状态：
112|
113|- **ConversationBufferMemory**：存储完整对话历史
114|- **ConversationSummaryMemory**：存储对话摘要以节省空间
115|- **VectorStoreRetrieverMemory**：使用向量存储实现的语义记忆
116|
117|示例：
118|```python
119|from langchain.memory import ConversationBufferMemory
120|from langchain.chains import ConversationChain
121|
122|memory = ConversationBufferMemory()
123|conversation = ConversationChain(
124|    llm=ChatOpenAI(),
125|    memory=memory,
126|    verbose=True
127|)
128|
129|conversation.predict(input="你好，我叫王小明")
130|conversation.predict(input="你记得我的名字吗？")
131|```
132|
133|### 2.4 检索 (Retrievers)
134|
135|检索组件用于从各种数据源获取相关信息：
136|
137|- **向量存储检索**：基于语义相似度检索文档
138|- **多查询检索**：使用多个不同查询增强检索结果
139|- **上下文压缩检索**：删减不相关内容以优化上下文窗口
140|
141|示例：
142|```python
143|from langchain.vectorstores import FAISS
144|from langchain.embeddings import OpenAIEmbeddings
145|from langchain.retrievers import ContextualCompressionRetriever
146|from langchain.retrievers.document_compressors import LLMChainExtractor
147|
148|# 创建向量存储
149|embeddings = OpenAIEmbeddings()
150|vectorstore = FAISS.from_texts(["内容1", "内容2", "内容3"], embeddings)
151|
152|# 基本检索器
153|retriever = vectorstore.as_retriever()
154|
155|# 上下文压缩检索器
156|compressor = LLMChainExtractor.from_llm(ChatOpenAI())
157|compression_retriever = ContextualCompressionRetriever(
158|    base_compressor=compressor,
159|    base_retriever=retriever
160|)
161|```
162|
163|### 2.5 输出解析器 (Output Parsers)
164|
165|输出解析器将语言模型的输出转换为结构化的格式：
166|
167|- **PydanticOutputParser**：解析为 Pydantic 模型
168|- **StrOutputParser**：提取纯文本输出
169|- **JsonOutputParser**：解析 JSON 格式输出
170|
171|示例：
172|```python
173|from langchain.output_parsers import PydanticOutputParser
174|from pydantic import BaseModel, Field
175|from typing import List
176|
177|class Movie(BaseModel):
178|    title: str = Field(description="电影标题")
179|    director: str = Field(description="导演姓名")
180|    year: int = Field(description="上映年份")
181|
182|class MovieList(BaseModel):
183|    movies: List[Movie] = Field(description="电影列表")
184|
185|parser = PydanticOutputParser(pydantic_object=MovieList)
186|```
187|
188|## 3. 单智能体系统
189|
190|### 3.1 智能体架构
191|
192|智能体系统是 LangChain 中的高级组件，允许语言模型使用工具并进行推理。智能体架构包括：
193|
194|- **智能体 (Agent)**：决定下一步行动的语言模型
195|- **工具 (Tools)**：智能体可以使用的函数
196|- **执行器 (AgentExecutor)**：协调智能体与工具之间的交互
197|
198|### 3.2 主要智能体类型
199|
200|#### 3.2.1 ReAct 智能体
201|
202|ReAct（Reasoning + Acting）智能体结合了推理和行动的能力，是最常用的智能体类型之一。
203|
204|**特点：**
205|- 结合推理（思考）和行动的能力
206|- 提供中间推理步骤
207|- 支持"链式思考"过程
208|
209|```python
210|from langchain.agents import AgentType, initialize_agent, Tool
211|from langchain.chat_models import ChatOpenAI
212|from langchain.memory import ConversationBufferMemory
213|from langchain_community.utilities import WikipediaAPIWrapper
214|
215|# 创建工具
216|wikipedia = WikipediaAPIWrapper()
217|tools = [Tool(name="维基百科", func=wikipedia.run, description="用于查询信息的工具")]
218|
219|# 创建LLM
220|llm = ChatOpenAI(temperature=0)
221|
222|# 创建记忆组件
223|memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
224|
225|# 初始化ReAct智能体
226|agent = initialize_agent(
227|    tools, 
228|    llm, 
229|    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
230|    verbose=True,
231|    memory=memory
232|)
233|```
234|
235|#### 3.2.2 OpenAI函数智能体
236|
237|OpenAI函数智能体利用OpenAI模型的函数调用能力，提供更结构化的工具使用方式。
238|
239|**特点：**
240|- 基于OpenAI的函数调用API
241|- 工具调用更加可靠
242|- 减少解析错误和幻觉
243|
244|```python
245|from langchain.agents import AgentType, initialize_agent, Tool
246|from langchain.chat_models import ChatOpenAI
247|
248|# 创建工具
249|tools = [Tool(name="计算器", func=lambda x: eval(x), description="用于数学计算")]
250|
251|# 创建OpenAI函数智能体
252|llm = ChatOpenAI(temperature=0)
253|agent = initialize_agent(
254|    tools,
255|    llm,
256|    agent=AgentType.OPENAI_FUNCTIONS,
257|    verbose=True
258|)
259|```

001|## 4. 多智能体系统
002|
003|### 4.1 多智能体编排模式
004|
005|#### 4.1.1 团队监督者模式
006|
007|团队监督者模式使用一个监督者智能体协调多个专家智能体，类似于团队领导与成员的关系。
008|
009|**核心特性:**
010|- 一个监督者智能体协调多个专家智能体
011|- 任务分解与分配
012|- 结果整合和协调
013|
014|```python
015|from langchain.agents import AgentType, initialize_agent, Tool
016|from langchain.chat_models import ChatOpenAI
017|from langchain.prompts import PromptTemplate
018|from langchain.chains import LLMChain
019|
020|# 创建专家智能体
021|researcher = initialize_agent(...) # 研究员智能体
022|coder = initialize_agent(...) # 编码员智能体
023|critic = LLMChain(...) # 评论员智能体
024|
025|# 创建监督者
026|supervisor_prompt = PromptTemplate(template="...", input_variables=[...])
027|supervisor = LLMChain(llm=ChatOpenAI(), prompt=supervisor_prompt)
028|
029|# 执行工作流
030|def run_workflow(task):
031|    # 1. 监督者制定计划
032|    plan = supervisor.run(task=task)
033|    
034|    # 2. 研究员收集信息
035|    research_result = researcher.run(task)
036|    
037|    # 3. 编码员实现解决方案
038|    code_solution = coder.run(f"{task}\n{research_result}")
039|    
040|    # 4. 评论员评估解决方案
041|    critique = critic.run(solution=code_solution)
042|    
043|    # 5. 监督者整合结果
044|    final_solution = supervisor.run(results=[research_result, code_solution, critique])
045|    
046|    return final_solution
047|```
048|
049|#### 4.1.2 经理-工人模式
050|
051|经理-工人模式中，一个经理智能体分配和协调任务，多个工人智能体执行具体工作。
052|
053|**核心特性:**
054|- 层级结构的任务分配
055|- 经理智能体负责规划和监督
056|- 工人智能体负责执行具体任务
057|
058|```python
059|# 经理-工人模式需要自定义实现，基本逻辑如下：
060|
061|# 1. 创建经理智能体
062|manager_agent = create_manager_agent()
063|
064|# 2. 创建多个工人智能体
065|worker_agents = create_worker_agents()
066|
067|# 3. 实现工作流
068|def manager_worker_workflow(task):
069|    # 经理分解任务
070|    subtasks = manager_agent.plan_task(task)
071|    
072|    # 分配任务给工人
073|    results = {}
074|    for subtask in subtasks:
075|        worker = select_appropriate_worker(subtask, worker_agents)
076|        results[subtask.id] = worker.execute(subtask)
077|    
078|    # 经理整合结果
079|    final_result = manager_agent.integrate_results(results)
080|    
081|    return final_result
082|```
083|
084|#### 4.1.3 计划执行模式
085|
086|计划执行模式中，一个智能体负责规划，另一个负责执行，适用于需要精确规划的复杂任务。
087|
088|**核心特性:**
089|- 明确的规划-执行分离
090|- 规划者智能体设计详细计划
091|- 执行者智能体负责实施计划
092|
093|```python
094|from langchain.chat_models import ChatOpenAI
095|from langchain.prompts import PromptTemplate
096|from langchain.chains import LLMChain
097|from langchain.agents import initialize_agent
098|
099|# 创建规划者智能体
100|planner_prompt = PromptTemplate(template="...", input_variables=["task"])
101|planner = LLMChain(llm=ChatOpenAI(), prompt=planner_prompt)
102|
103|# 创建执行者智能体
104|executor = initialize_agent(...)
105|
106|# 执行计划-执行模式
107|def plan_execute_workflow(task):
108|    # 1. 制定计划
109|    plan = planner.run(task=task)
110|    
111|    # 2. 执行计划
112|    execution_result = executor.run(f"根据以下计划执行任务：\n任务: {task}\n计划:\n{plan}")
113|    
114|    return execution_result
115|```
116|
117|### 4.2 多智能体通信协议
118|
119|多智能体系统中，智能体之间需要有效沟通：
120|
121|- **消息格式化**：定义结构化消息格式
122|- **状态共享**：共享任务状态和进度
123|- **冲突解决**：处理不同智能体之间的意见分歧
124|
125|示例 (使用消息格式):
126|
127|```python
128|from typing import Dict, List, Any
129|
130|class Message:
131|    def __init__(self, sender: str, receiver: str, content: str, message_type: str, metadata: Dict[str, Any] = None):
132|        self.sender = sender
133|        self.receiver = receiver
134|        self.content = content
135|        self.message_type = message_type  # request, response, update等
136|        self.metadata = metadata or {}
137|
138|# 消息传递函数
139|def send_message(message: Message, agents: Dict[str, Any]):
140|    if message.receiver in agents:
141|        return agents[message.receiver].process_message(message)
142|    return None
143|```
144|
145|## 5. 搜索与智能体集成
146|
147|### 5.1 检索增强生成（RAG）与智能体结合
148|
149|RAG系统可以增强智能体的知识，为其提供更多相关信息。
150|
151|**实现方法:**
152|
153|#### 5.1.1 RAG作为工具
154|
155|智能体可以将RAG系统作为一个工具来使用：
156|
157|```python
158|from langchain.document_loaders import TextLoader
159|from langchain.text_splitter import RecursiveCharacterTextSplitter
160|from langchain.embeddings import OpenAIEmbeddings
161|from langchain.vectorstores import FAISS
162|from langchain.agents import AgentType, initialize_agent, Tool
163|from langchain.chat_models import ChatOpenAI
164|
165|# 创建RAG系统
166|documents = loader.load()
167|text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
168|chunks = text_splitter.split_documents(documents)
169|vectorstore = FAISS.from_documents(chunks, OpenAIEmbeddings())
170|retriever = vectorstore.as_retriever()
171|
172|# 创建RAG查询工具
173|def query_knowledge_base(query):
174|    docs = retriever.get_relevant_documents(query)
175|    return "\n\n".join(doc.page_content for doc in docs)
176|
177|# 将RAG作为智能体工具
178|rag_tool = Tool(
179|    name="知识库查询",
180|    func=query_knowledge_base,
181|    description="当你需要查询专业知识时使用此工具"
182|)
183|
184|# 创建使用RAG工具的智能体
185|agent = initialize_agent(
186|    tools=[rag_tool],
187|    llm=ChatOpenAI(),
188|    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
189|    verbose=True
190|)
191|```
192|
193|#### 5.1.2 上下文增强
194|
195|在智能体处理之前使用RAG增强提示：
196|
197|```python
198|# 先检索相关文档
199|query = "量子计算的应用场景?"
200|docs = retriever.get_relevant_documents(query)
201|context = "\n\n".join(doc.page_content for doc in docs)
202|
203|# 创建增强的提示
204|enhanced_prompt = f"基于以下信息回答问题。\n\n信息: {context}\n\n问题: {query}"
205|
206|# 创建智能体并使用增强提示
207|agent = initialize_agent(...)
208|result = agent.run(enhanced_prompt)
209|```
210|
211|### 5.2 高级搜索技术
212|
213|#### 5.2.1 混合检索
214|
215|混合检索结合多种检索策略，提高相关性和准确性：
216|
217|```python
218|from langchain.retrievers import EnsembleRetriever
219|from langchain.vectorstores import FAISS
220|from langchain_community.retrievers import BM25Retriever
221|
222|# 创建向量检索器
223|vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
224|
225|# 创建关键词检索器
226|keyword_retriever = BM25Retriever.from_documents(documents)
227|
228|# 创建集成检索器
229|ensemble_retriever = EnsembleRetriever(
230|    retrievers=[vector_retriever, keyword_retriever],
231|    weights=[0.7, 0.3]
232|)
233|
234|# 获取混合检索结果
235|docs = ensemble_retriever.get_relevant_documents(query)
236|```
237|
238|#### 5.2.2 上下文压缩
239|
240|上下文压缩技术可以优化检索结果，提取最相关内容：
241|
242|```python
243|from langchain.retrievers import ContextualCompressionRetriever
244|from langchain.retrievers.document_compressors import LLMChainExtractor
245|
246|# 创建基本检索器
247|retriever = vectorstore.as_retriever()
248|
249|# 创建压缩器
250|compressor = LLMChainExtractor.from_llm(ChatOpenAI())
251|
252|# 创建上下文压缩检索器
253|compression_retriever = ContextualCompressionRetriever(
254|    base_compressor=compressor,
255|    base_retriever=retriever
256|)
257|
258|# 获取压缩后的检索结果
259|compressed_docs = compression_retriever.get_relevant_documents(query)
260|```
261|
262|#### 5.2.3 查询重写
263|
264|使用语言模型改进原始查询以提高检索质量：
265|
266|```python
267|from langchain.chains import LLMChain
268|from langchain.prompts import PromptTemplate
269|
270|# 创建查询重写器
271|rewriter_prompt = PromptTemplate(
272|    input_variables=["query"],
273|    template="请将以下查询重写为更有效的搜索查询，以便于在知识库中检索信息:\n\n{query}\n\n改进后的查询:"
274|)
275|rewriter = LLMChain(llm=ChatOpenAI(), prompt=rewriter_prompt)
276|
277|# 重写查询
278|original_query = "量子计算机什么时候能实用化?"
279|improved_query = rewriter.run(original_query)
280|
281|# 使用改进后的查询
282|docs = retriever.get_relevant_documents(improved_query)
283|```
284|
285|## 6. 多智能体工作流设计模式
286|
287|### 6.1 研究-规划-执行模式
288|
289|这种模式将工作流分为三个阶段：研究、规划和执行，由不同专长的智能体负责。
290|
291|**工作流程:**
292|1. **研究阶段**: 研究智能体收集和分析信息
293|2. **规划阶段**: 规划智能体基于研究结果制定计划
294|3. **执行阶段**: 执行智能体实施计划并完成任务
295|
296|```python
297|from langchain.agents import initialize_agent
298|from langchain.chat_models import ChatOpenAI
299|from langchain.prompts import PromptTemplate
300|from langchain.chains import LLMChain
301|
302|# 创建研究智能体
303|research_agent = initialize_agent(
304|    [search_tool, knowledge_tool],
305|    ChatOpenAI(temperature=0),
306|    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
307|    verbose=True
308|)
309|
310|# 创建规划智能体
311|planner_prompt = PromptTemplate(
312|    template="基于以下研究结果，为解决{task}制定详细计划:\n\n{research_results}\n\n详细计划:",
313|    input_variables=["task", "research_results"]
314|)
315|planner = LLMChain(llm=ChatOpenAI(temperature=0), prompt=planner_prompt)
316|
317|# 创建执行智能体
318|execution_agent = initialize_agent(
319|    [code_tool, calculator_tool, api_tool],
320|    ChatOpenAI(temperature=0),
321|    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
322|    verbose=True
323|)
324|
325|# 执行工作流
326|def research_plan_execute_workflow(task):
327|    # 1. 研究阶段
328|    research_results = research_agent.run(f"收集关于{task}的所有必要信息")
329|    
330|    # 2. 规划阶段
331|    plan = planner.run(task=task, research_results=research_results)
332|    
333|    # 3. 执行阶段
334|    result = execution_agent.run(f"按照以下计划执行任务:\n\nTask: {task}\n\n计划:\n{plan}\n\n研究信息:\n{research_results}")
335|    
336|    return result
337|```
338|
339|### 6.2 分析-创建-评估模式
340|
341|这种模式适合创造性任务，包含问题分析、创建解决方案和评估改进三个阶段。
342|
343|**工作流程:**
344|1. **分析阶段**: 分析智能体理解问题和需求
345|2. **创建阶段**: 创造智能体生成解决方案
346|3. **评估阶段**: 评估智能体检查和改进解决方案
347|
348|```python
349|from langchain.chat_models import ChatOpenAI
350|from langchain.chains import LLMChain
351|from langchain.prompts import PromptTemplate
352|
353|# 创建分析智能体
354|analysis_prompt = PromptTemplate(
355|    template="深入分析以下问题，确定关键需求、约束条件和成功标准:\n\n{problem}\n\n详细分析:",
356|    input_variables=["problem"]
357|)
358|analyzer = LLMChain(llm=ChatOpenAI(temperature=0), prompt=analysis_prompt)
359|
360|# 创建创造智能体
361|creation_prompt = PromptTemplate(
362|    template="基于以下分析，为问题{problem}创建解决方案:\n\n分析:\n{analysis}\n\n创新解决方案:",
363|    input_variables=["problem", "analysis"]
364|)
365|creator = LLMChain(llm=ChatOpenAI(temperature=0.7), prompt=creation_prompt)
366|
367|# 创建评估智能体
368|evaluation_prompt = PromptTemplate(
369|    template="评估以下解决方案，指出优点、缺点和改进建议:\n\n问题: {problem}\n\n分析:\n{analysis}\n\n解决方案:\n{solution}\n\n详细评估:",
370|    input_variables=["problem", "analysis", "solution"]
371|)
372|evaluator = LLMChain(llm=ChatOpenAI(temperature=0), prompt=evaluation_prompt)
373|
374|# 执行工作流
375|def analyze_create_evaluate_workflow(problem):
376|    # 1. 分析阶段
377|    analysis = analyzer.run(problem=problem)
378|    
379|    # 2. 创建阶段
380|    solution = creator.run(problem=problem, analysis=analysis)
381|    
382|    # 3. 评估阶段
383|    evaluation = evaluator.run(problem=problem, analysis=analysis, solution=solution)
384|    
385|    # 4. 返回完整结果
386|    return {
387|        "analysis": analysis,
388|        "solution": solution,
389|        "evaluation": evaluation
390|    }
391|```
392|
393|### 6.3 协作迭代模式
394|
395|多个智能体并行工作，定期同步和迭代改进解决方案。
396|
397|**工作流程:**
398|1. **问题分解**: 将任务分解为可并行处理的部分
399|2. **并行工作**: 多个智能体同时处理不同子任务
400|3. **同步协调**: 定期交流和整合进展
401|4. **迭代改进**: 基于反馈不断调整和优化
402|
403|```python
404|from langchain.chat_models import ChatOpenAI
405|from langchain.chains import LLMChain
406|from langchain.prompts import PromptTemplate
407|from concurrent.futures import ThreadPoolExecutor
408|
409|# 创建任务分解器
410|decomposer_prompt = PromptTemplate(
411|    template="将以下任务分解为可以并行处理的子任务:\n\n任务: {task}\n\n子任务列表 (JSON格式):",
412|    input_variables=["task"]
413|)
414|decomposer = LLMChain(llm=ChatOpenAI(temperature=0), prompt=decomposer_prompt)
415|
416|# 创建协调器
417|coordinator_prompt = PromptTemplate(
418|    template="基于以下子任务的处理结果，协调并整合信息:\n\n子任务结果:\n{subtask_results}\n\n整合后的结果:",
419|    input_variables=["subtask_results"]
420|)
421|coordinator = LLMChain(llm=ChatOpenAI(temperature=0), prompt=coordinator_prompt)
422|
423|# 创建工作者（可使用不同的专业智能体）
424|def create_worker(worker_name):
425|    worker_prompt = PromptTemplate(
426|        template="你是专门负责{worker_name}的专家。处理以下子任务并给出结果:\n\n{subtask}\n\n详细结果:",
427|        input_variables=["worker_name", "subtask"]
428|    )
429|    return LLMChain(llm=ChatOpenAI(temperature=0.2), prompt=worker_prompt)
430|
431|# 执行协作迭代工作流
432|def collaborative_workflow(task, num_iterations=3):
433|    # 分解任务
434|    subtasks_json = decomposer.run(task=task)
435|    subtasks = json.loads(subtasks_json)
436|    
437|    # 创建专家工作者
438|    workers = {
439|        "研究": create_worker("研究"),
440|        "设计": create_worker("设计"),
441|        "实现": create_worker("实现"),
442|        "测试": create_worker("测试")
443|    }
444|    
445|    # 执行多轮迭代
446|    results = {}
447|    for iteration in range(num_iterations):
448|        print(f"执行第 {iteration+1} 轮迭代")
449|        
450|        # 并行处理子任务
451|        iteration_results = {}
452|        with ThreadPoolExecutor() as executor:
453|            futures = {}
454|            for i, subtask in enumerate(subtasks):
455|                worker_name = list(workers.keys())[i % len(workers)]
456|                futures[executor.submit(workers[worker_name].run, worker_name=worker_name, subtask=subtask)] = i
457|            
458|            for future in futures:
459|                idx = futures[future]
460|                iteration_results[idx] = future.result()
461|        
462|        # 协调和整合结果
463|        subtask_results = "\n\n".join([f"子任务 {k}: {v}" for k, v in iteration_results.items()])
464|        if iteration < num_iterations - 1:
465|            feedback = coordinator.run(subtask_results=subtask_results)
466|            print(f"第 {iteration+1} 轮反馈: {feedback}")
467|            # 更新子任务，加入反馈
468|            subtasks = [f"{subtask}\n\n前一轮反馈: {feedback}" for subtask in subtasks]
469|        else:
470|            # 最后一轮，生成最终结果
471|            final_result = coordinator.run(subtask_results=subtask_results)
472|            results = final_result
473|    
474|    return results
475|```
476|
477|## 7. 最佳实践与优化技巧
478|
479|### 7.1 智能体设计原则
480|
481|- **明确角色和职责**: 每个智能体应有明确的专长和任务范围
482|- **提供足够上下文**: 智能体需要充分的背景信息来做出决策
483|- **设计适当的提示**: 提示模板直接影响智能体的效果
484|- **错误处理机制**: 添加错误检测和恢复机制
485|- **循环避免**: 防止智能体陷入无限循环
486|
487|### 7.2 多智能体协作优化
488|
489|- **有效的通信协议**: 定义智能体间交流的格式和规则
490|- **任务分解粒度**: 适当的任务分解粒度可提高效率
491|- **结果整合机制**: 设计合理的结果整合和冲突解决方案
492|- **监控和干预**: 实现监控机制，必要时允许人类干预
493|
494|### 7.3 性能优化
495|
496|- **批量处理**: 尽可能批量处理查询以减少API调用
497|- **缓存机制**: 缓存常用查询和中间结果
498|- **异步处理**: 使用异步API减少等待时间
499|- **选择合适的模型**: 根据任务复杂度选择合适的底层模型
500|
501|## 8. 结论
502|
503|LangChain提供了一个强大的框架，用于构建基于大语言模型的应用程序。通过组合模型、提示工程、记忆系统、检索组件和输出解析器等核心组件，开发者可以创建功能丰富的智能应用。从单一智能体到复杂的多智能体编排，从基本的问答系统到搜索增强的数据处理系统，LangChain提供了灵活而强大的工具集。
504|
505|随着框架的不断发展，我们可以期待更多的组件、更高效的编排模式和更强大的功能。通过理解和应用本文介绍的各种模式和技术，开发者可以充分发挥大语言模型的潜力，构建解决实际问题的智能应用。
506|
507|## 参考资料
508|
509|- LangChain 官方文档: https://python.langchain.com/docs/introduction/
510|- LangChain GitHub 仓库: https://github.com/langchain-ai/langchain
511|- LangChain 示例集: https://python.langchain.com/docs/use_cases/
512|- LangChain 集成: https://python.langchain.com/docs/integrations/