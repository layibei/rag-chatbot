drop table if exists index_log;
create table if not exists index_log (
    id bigserial primary key,
    source varchar(255) not null,
    checksum varchar(255) not null,
    indexed_time timestamp not null,
    indexed_by varchar(128) not null,
    modified_time timestamp not null,
    modified_by varchar(128) not null,
    constraint unique_index_log unique (source, checksum)
);

drop index if exists idx_source_checksum;
create index if not exists idx_source_checksum on index_log (source, checksum);