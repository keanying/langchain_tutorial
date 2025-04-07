#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
API密钥管理模块

本模块用于管理LangChain应用中使用的各种API密钥，使用环境变量进行安全管理。
模块提供了以下功能：
1. 配置各种AI服务提供商的API密钥
2. 检查必需的API密钥是否已设置
3. 提供获取API密钥的安全方法
"""

import os
from typing import Dict, List, Optional
from dotenv import load_dotenv
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()

# 定义支持的API密钥和相关服务
SUPPORTED_APIS = {
    'OPENAI_API_KEY': {
        'description': 'OpenAI API密钥，用于访问GPT-3.5, GPT-4等模型',
        'required_for': ['OpenAI', 'ChatOpenAI', 'OpenAIEmbeddings'],
        'url': 'https://platform.openai.com/account/api-keys'
    },
    'ANTHROPIC_API_KEY': {
        'description': 'Anthropic API密钥，用于访问Claude系列模型',
        'required_for': ['ChatAnthropic', 'Anthropic'],
        'url': 'https://console.anthropic.com/account/keys'
    },
    'HUGGINGFACEHUB_API_TOKEN': {
        'description': 'Hugging Face Hub API令牌，用于访问Hugging Face的模型',
        'required_for': ['HuggingFaceHub', 'HuggingFaceEmbeddings'],
        'url': 'https://huggingface.co/settings/tokens'
    },
    'COHERE_API_KEY': {
        'description': 'Cohere API密钥，用于访问Cohere的模型',
        'required_for': ['Cohere', 'CohereEmbeddings'],
        'url': 'https://dashboard.cohere.ai/api-keys'
    },
    'SERPAPI_API_KEY': {
        'description': 'SerpAPI密钥，用于网络搜索功能',
        'required_for': ['SerpAPIWrapper'],
        'url': 'https://serpapi.com/manage-api-key'
    },
    'GOOGLE_API_KEY': {
        'description': 'Google API密钥，用于Google搜索和其他Google服务',
        'required_for': ['GoogleSearchAPIWrapper', 'GoogleCustomSearch'],
        'url': 'https://console.cloud.google.com/apis/credentials'
    },
    'GOOGLE_CSE_ID': {
        'description': 'Google自定义搜索引擎ID',
        'required_for': ['GoogleCustomSearch'],
        'url': 'https://programmablesearchengine.google.com/controlpanel/all'
    }
}


def get_api_key(key_name: str) -> Optional[str]:
    """
    获取指定的API密钥
    
    Args:
        key_name: API密钥的名称，如'OPENAI_API_KEY'
        
    Returns:
        如果密钥已设置，则返回密钥值，否则返回None
    """
    api_key = os.environ.get(key_name)
    if api_key is None or api_key.strip() == "":
        logger.warning(f"环境变量 {key_name} 未设置")
        return None
    return api_key


def check_api_key(key_name: str) -> bool:
    """
    检查指定的API密钥是否已设置
    
    Args:
        key_name: API密钥的名称
        
    Returns:
        如果密钥已设置，则返回True，否则返回False
    """
    return get_api_key(key_name) is not None


def check_required_keys(component_name: str) -> Dict[str, bool]:
    """
    检查指定组件所需的所有API密钥是否已设置
    
    Args:
        component_name: 组件名称，如'OpenAI'或'ChatAnthropic'
        
    Returns:
        包含每个必需密钥及其状态的字典
    """
    required_keys = {}
    
    for key, info in SUPPORTED_APIS.items():
        if component_name in info['required_for']:
            required_keys[key] = check_api_key(key)
    
    return required_keys


def get_missing_keys(component_name: str) -> List[str]:
    """
    获取指定组件缺少的API密钥列表
    
    Args:
        component_name: 组件名称
        
    Returns:
        缺少的API密钥名称列表
    """
    required_keys = check_required_keys(component_name)
    return [key for key, is_set in required_keys.items() if not is_set]


def set_api_key(key_name: str, key_value: str) -> None:
    """
    设置指定的API密钥（仅在当前会话中有效）
    
    Args:
        key_name: API密钥的名称
        key_value: API密钥的值
    """
    os.environ[key_name] = key_value
    logger.info(f"已设置环境变量 {key_name}")


def print_api_key_status() -> None:
    """打印所有支持的API密钥的状态"""
    print("\n=== API密钥状态 ===\n")
    
    for key, info in SUPPORTED_APIS.items():
        status = "✅ 已设置" if check_api_key(key) else "❌ 未设置"
        print(f"{key}: {status}")
        print(f"  描述: {info['description']}")
        print(f"  获取地址: {info['url']}")
        print(f"  必需组件: {', '.join(info['required_for'])}")
        print()


def setup_required_keys_for(component_name: str) -> bool:
    """
    检查并提示用户设置指定组件所需的API密钥
    
    Args:
        component_name: 组件名称
    
    Returns:
        如果所有必需的密钥都已设置，则返回True，否则返回False
    """
    missing_keys = get_missing_keys(component_name)
    
    if not missing_keys:
        logger.info(f"{component_name}所需的所有API密钥已设置")
        return True
    
    logger.warning(f"{component_name}缺少以下API密钥：{', '.join(missing_keys)}")
    for key in missing_keys:
        info = SUPPORTED_APIS.get(key, {})
        logger.info(f"请访问 {info.get('url', '官方网站')} 获取 {key}")
    
    return False


if __name__ == "__main__":
    print_api_key_status()