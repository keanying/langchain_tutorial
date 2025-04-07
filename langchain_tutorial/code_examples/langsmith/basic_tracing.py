"""
LangSmith基本追踪示例

本示例展示如何使用LangSmith跟踪LangChain应用程序的执行过程，
包括基本设置、自定义跟踪和查询追踪数据。
"""

import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.tracers import LangSmithTracer
from langsmith import Client

# 设置LangSmith环境变量
os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-api-key"  # 请替换为您的LangSmith API密钥
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "langsmith-demo"

def basic_tracing_example():
    """基本追踪示例：无需额外代码，自动跟踪LangChain组件"""
    print("🔍 执行基本追踪示例...")
    
    # 创建LLM实例
    llm = ChatOpenAI(temperature=0.7)
    
    # 创建提示模板
    template = "你是一位{role}专家。请简要回答以下问题：{question}"
    prompt = ChatPromptTemplate.from_template(template)
    
    # 创建链
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # 执行链 - 会自动追踪
    response = chain.invoke({
        "role": "人工智能",
        "question": "机器学习和深度学习有什么区别？"
    })
    
    print(f"回答: {response['text']}")
    print("✅ 运行已自动追踪到LangSmith。请登录LangSmith控制台查看详情。")
    
    return response

def custom_tracing_example():
    """自定义追踪示例：使用自定义标签和元数据"""
    print("\n🔍 执行自定义追踪示例...")
    
    # 创建LLM实例
    llm = ChatOpenAI(temperature=0.7)
    
    # 创建提示模板
    template = "作为{domain}方面的专家，请详细解释{topic}概念，并提供一个简单的例子。"
    prompt = ChatPromptTemplate.from_template(template)
    
    # 创建链
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # 创建自定义跟踪器
    tracer = LangSmithTracer(
        project_name="custom-traces",
        tags=["教育内容", "技术解释"],
        metadata={
            "user_id": "user-123",
            "session_id": "session-456",
            "request_source": "tutorial_demo",
            "complexity_level": "intermediate"
        }
    )
    
    # 执行链 - 使用自定义跟踪器
    response = chain.invoke(
        {
            "domain": "计算机科学",
            "topic": "递归算法"
        },
        config={"callbacks": [tracer]}
    )
    
    print(f"回答: {response['text'][:100]}...")
    print("✅ 运行已使用自定义元数据追踪到LangSmith。")
    
    return response

def manual_tracing_example():
    """手动追踪示例：追踪非LangChain组件或自定义处理步骤"""
    print("\n🔍 执行手动追踪示例...")
    
    # 创建LangSmith客户端
    client = Client()
    
    # 模拟一个自定义处理函数
    def process_data(input_text):
        # 手动创建根跟踪项
        with client.trace(
            name="自定义数据处理流程",
            run_type="chain",
            project_name="manual-traces",
            tags=["data-processing"],
            metadata={"input_length": len(input_text)}
        ) as run:
            try:
                # 记录接收到的数据
                run.add_inputs({"raw_input": input_text})
                
                # 模拟第一个处理步骤
                with run.trace("步骤1：分词", run_type="tool") as step1:
                    # 模拟一些处理逻辑
                    tokens = input_text.split()
                    step1.add_outputs({"tokens": tokens, "token_count": len(tokens)})
                
                # 模拟第二个处理步骤
                with run.trace("步骤2：过滤停用词", run_type="tool") as step2:
                    # 模拟一些处理逻辑
                    stopwords = ["的", "了", "是"]
                    filtered_tokens = [token for token in tokens if token not in stopwords]
                    step2.add_outputs({"filtered_tokens": filtered_tokens})
                
                # 模拟最终结果
                result = " ".join(filtered_tokens)
                run.add_outputs({"processed_result": result})
                
                return result
                
            except Exception as e:
                # 记录错误
                run.add_outputs({"error": str(e)})
                run.end(error=str(e))
                raise
    
    # 调用自定义处理函数
    result = process_data("这是一个用于演示手动追踪功能的示例文本，它将被处理并记录到LangSmith中。")
    
    print(f"处理结果: {result}")
    print("✅ 自定义处理步骤已手动追踪到LangSmith。")
    
    return result

def query_runs_example():
    """查询追踪数据示例：获取和分析先前的运行记录"""
    print("\n🔍 执行追踪数据查询示例...")
    
    try:
        # 创建LangSmith客户端
        client = Client()
        
        # 查询最近的运行记录
        runs = list(client.list_runs(
            project_name="langsmith-demo",
            execution_order=1,  # 按执行时间排序
            limit=5  # 限制返回数量
        ))
        
        if not runs:
            print("未找到运行记录。请先运行其他示例以生成一些跟踪数据。")
            return
        
        print(f"找到 {len(runs)} 条最近的运行记录:")
        
        for i, run in enumerate(runs):
            print(f"\n运行 {i+1}:")
            print(f"  ID: {run.id}")
            print(f"  名称: {run.name}")
            print(f"  状态: {run.status}")
            print(f"  运行时间: {run.latency_ms if run.latency_ms else 'N/A'} 毫秒")
            print(f"  输入: {list(run.inputs.keys()) if run.inputs else 'N/A'}")
            print(f"  输出: {list(run.outputs.keys()) if run.outputs else 'N/A'}")
            if run.error:
                print(f"  错误: {run.error}")
        
        # 计算平均延迟
        latencies = [run.latency_ms for run in runs if run.latency_ms]
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            print(f"\n平均延迟: {avg_latency:.2f} 毫秒")
        
        print("✅ 成功查询并分析了LangSmith中的运行记录。")
        
        return runs
        
    except Exception as e:
        print(f"查询运行记录时出错: {str(e)}")
        return None

def main():
    """主函数，运行所有示例"""
    print("===== LangSmith 基本追踪示例 =====\n")
    
    # 检查环境变量
    if os.environ.get("LANGCHAIN_API_KEY") == "your-langsmith-api-key":
        print("⚠️ 警告: 请将代码中的'your-langsmith-api-key'替换为您的实际LangSmith API密钥")
        print("申请API密钥: https://smith.langchain.com/\n")
    
    # 运行示例
    basic_tracing_example()
    custom_tracing_example()
    manual_tracing_example()
    query_runs_example()
    
    print("\n===== 所有示例完成 =====")
    print("登录 https://smith.langchain.com/ 查看详细的追踪结果")

if __name__ == "__main__":
    main()