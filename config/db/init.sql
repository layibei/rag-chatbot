drop table if exists index_logs;
drop index if exists unique_index_log;
create table if not exists index_logs (
    id bigserial primary key,
    source varchar(1024) not null,
    source_type varchar(128) not null,
    checksum varchar(255) not null,
    created_at timestamp not null,
    created_by varchar(128) not null,
    modified_at timestamp not null,
    modified_by varchar(128) not null,
    status varchar(128) not null,
    error_message text,
    constraint uix_source_source_type unique (source, source_type),
    constraint uix_checksum unique (checksum)
);

drop index if exists idx_source_checksum;
create index if not exists idx_source_checksum on index_logs (source, source_type, checksum);