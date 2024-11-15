# rag-chatbot
Use langchain series to build a simple RAG solution.

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
```

# Index logs table in postgres
```sql
drop table if exists index_logs;
drop index if exists unique_index_log;
create table if not exists index_logs (
    id bigserial primary key,
    source varchar(512) not null,
    checksum varchar(255) not null,
    indexed_time timestamp not null,
    indexed_by varchar(128) not null,
    modified_time timestamp not null,
    modified_by varchar(128) not null,
    status varchar(128) not null,
    constraint unique_index_log unique (source, checksum)
);

drop index if exists idx_source_checksum;
create index if not exists idx_source_checksum on index_logs (source, checksum);
```