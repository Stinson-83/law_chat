from __future__ import annotations
from sqlalchemy import (
    Column, Integer, BigInteger, Text, JSON, ForeignKey, text as sqltext,
    TIMESTAMP
)
from sqlalchemy.orm import declarative_base, relationship
from pgvector.sqlalchemy import Vector

Base = declarative_base()

class DocRaw(Base):
    __tablename__ = 'docs_raw'
    id = Column(BigInteger, primary_key=True)
    filename = Column(Text)
    title = Column(Text)
    year = Column(Integer)
    category = Column(Text)
    data = Column(JSON, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=sqltext('now()'))
    updated_at = Column(TIMESTAMP(timezone=True), server_default=sqltext('now()'))
    passages = relationship('Passage', back_populates='doc', cascade='all, delete')

class Passage(Base):
    __tablename__ = 'passages'
    id = Column(BigInteger, primary_key=True)
    doc_id = Column(BigInteger, ForeignKey('docs_raw.id', ondelete='CASCADE'))
    section_no = Column(Text)
    heading = Column(Text)
    text = Column(Text, nullable=False)
    embedding = Column(Vector(768))
    year = Column(Integer)
    category = Column(Text)
    token_count = Column(Integer)
    checksum = Column(Text)
    created_at = Column(TIMESTAMP(timezone=True), server_default=sqltext('now()'))
    updated_at = Column(TIMESTAMP(timezone=True), server_default=sqltext('now()'))

    doc = relationship('DocRaw', back_populates='passages')