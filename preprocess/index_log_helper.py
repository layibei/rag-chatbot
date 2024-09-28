import traceback

from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import declarative_base, sessionmaker

from preprocess import IndexLog
from utils.logging_util import logger

Base = declarative_base()


class IndexLogHelper():
    def __init__(self, uri: str):
        self.logger = logger
        self.engine = create_engine(uri)
        Base.metadata.create_all(self.engine)

    def create_session(self):
        session = sessionmaker(bind=self.engine)
        return session()

    def save(self, index_log: IndexLog):
        if index_log.id is None:
            return

        with self.create_session() as session:
            try:
                session.add(index_log)
                session.commit()
                self.logger.info(f'Index log saved successfully, {index_log}')
            except SQLAlchemyError as e:
                session.rollback()
                self.logger.error(f'Error while saving index log,{traceback.format_exc()}')
                raise e

    def find_by_checksum(self, checksum: str):
        if checksum is None:
            return None

        with self.create_session() as session:
            try:
                log = session.query(IndexLog).filter(IndexLog.checksum == checksum).first()
            except SQLAlchemyError as e:
                self.logger.error(f'Error while finding index log for {checksum}, {traceback.format_exc()}')
                raise e

            return log

    def find_by_source(self, file_path: str):
        if file_path is None:
            return None

        with self.create_session() as session:
            try:
                log = session.query(IndexLog).filter(IndexLog.source == file_path).first()
                self.logger.info(f'Index log found for {file_path}, {log}')
            except SQLAlchemyError as e:
                self.logger.error(f'Error while finding index log for {file_path}, {traceback.format_exc()}')
                raise e

            return log

    def delete_by_source(self, file_path: str):
        if file_path is None:
            return None

        with self.create_session() as session:
            try:
                session.query(IndexLog).filter(IndexLog.source == file_path).delete()
                session.commit()
                self.logger.info(f'Index log deleted for {file_path}')
            except SQLAlchemyError as e:
                session.rollback()
                self.logger.error(f'Error while deleting index log for {file_path}, {traceback.format_exc()}')
                raise e
