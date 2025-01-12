# rag-chatbot
Use langchain series to build a simple RAG solution.
![rag-hl-flow.png](readme%2Frag-hl-flow.png)

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