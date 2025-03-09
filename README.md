# RAG Chatbot

一个基于 RAG (Retrieval Augmented Generation) 的问答系统。

## 项目结构

### 主要目录

- `api/` - API 路由和接口定义
  - `chat_routes.py` - 聊天相关的 API 端点
  - `health_routes.py` - 健康检查端点

- `config/` - 配置相关
  - `common_settings.py` - 通用配置，包括 LLM、数据库等设置
  - `database/` - 数据库配置
    - `database_manager.py` - 数据库连接管理

- `handler/` - 业务逻辑处理
  - `generic_query_handler.py` - 通用查询处理器，处理用户输入
  - `workflow/` - 工作流程定义
    - `query_process_workflow.py` - RAG 查询处理流程

- `preprocess/` - 数据预处理
  - `index_log.py` - 索引日志相关模型

- `tests/` - 测试代码
  - `conftest.py` - pytest 配置和通用 fixtures
  - `test_rag_evaluation.py` - RAG 系统评估测试
  - `data/` - 测试数据
    - `evaluation_dataset.json` - 评估数据集
  - `reports/` - 测试报告输出目录

### 数据存储

- **Qdrant DB**: 向量数据库，用于存储和检索文档向量
  - 用途：相似性搜索，RAG 的检索部分

- **PostgreSQL**: 关系型数据库
  - 用途：存储对话历史、索引日志、评估指标等

### 主要文件说明

- `main.py` - 应用入口点，FastAPI 服务器配置
- `requirements.txt` - 项目依赖
- `.env` - 环境变量配置
- `pytest.ini` - pytest 配置

## 核心功能

1. **RAG 检索增强生成**
   - 文档向量化和存储
   - 相似性检索
   - 上下文增强的回答生成

2. **对话管理**
   - 会话历史记录
   - 用户状态管理

3. **评估系统**
   - 回答质量评估
   - HTML 报告生成

## 配置说明

主要配置项：
- LLM 模型选择 (Google Gemini/Ollama)
- 向量数据库连接
- PostgreSQL 数据库连接
- API 密钥配置

## 测试

运行测试：

# support file formats
- pdf
- docx
- txt
- csv
- json


# add following keys to rag-chatbot/.env file - below are some sample key-values
![img.png](readme%2Fimg.png)
```shell
GOOGLE_API_KEY=AIzaSyAegq6kfNpiVxxchzeuFwmq7difvrc5239YX0  
HUGGINGFACEHUB_API_TOKEN=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  
SERPAPI_API_KEY=d43192bbae5c3dedd91525f9792dfghjkabc88ebb51c11661bdfb5ce86e6b33b  
LANGCHAIN_API_KEY=lsv2_pt_ac88b1c9c1bf4f6ertyui9d4159cac3_2a0317bd12  
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com  
LANGCHAIN_PROJECT=xxx  
LANGCHAIN_TRACING_V2=true
```

# use docker to run qdrant & pgvector & redis
- In app, use PG to do the first line check for files to be embedded, if already are indexed, then skip the embedding process,
Qdrant is still the vector store.
- Pgvector is also a vector database, will do some exploration on it.
```shell
docker run --name qdrant -e TZ=Etc/UTC -e RUN_MODE=production -v /d/Cloud/docker/volumes/qdrant/config:/qdrant/config -v /d/Cloud/docker/volumes/qdrant/data:/qdrant/storage -p 6333:6333 --restart=always  -d qdrant/qdrant


docker run -d \
  --name pgvector \
  -p 5432:5432 \
  -e POSTGRES_USER={POSTGRES_USER} \
  -e POSTGRES_PASSWORD={POSTGRES_PASSWORD} \
  -e POSTGRES_DB={POSTGRES_DB} \
  -v /d/Cloud/docker/volumes/pgvector/data:/var/lib/postgresql/data \
  ankane/pgvector
  
docker run \
    -d \
    --name=neo4j \
    --restart always \
    --publish=7474:7474 --publish=7687:7687 \
    --env NEO4J_AUTH=neo4j/{NEO4J_PASSWORD} \
    --volume=/D/Cloud/docker/volumes/neo4j:/data \
    neo4j:5.26.0
```

# Init the database
Run the init sql files under config/db

# Q&A
1.Too many free trial accounts used on this machine. Please upgrade to pro. We have this limit in place to prevent abuse. Please let us know if you believe this is a mistake. Request ID: 03916405-a2bf-4fe9-80e5-877001db1313
> Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process; iwr -useb https://raw.githubusercontent.com/resetsix/cursor_device_id/main/device_id_win.ps1 | iex