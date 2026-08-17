from sqlalchemy import Engine, MetaData, create_engine, text


def create_db_engine(database_url: str) -> Engine:
    """Build a SQLAlchemy engine for the given PostgreSQL connection URL.

    No global/module-level engine is created here -- callers own the
    engine's lifetime and pass it explicitly to repositories/indexers.
    """
    return create_engine(database_url, future=True)


def init_schema(engine: Engine, metadata: MetaData) -> None:
    """Create the pgvector extension (if missing) and all tables in
    `metadata`. Safe to call repeatedly -- both operations are
    idempotent (IF NOT EXISTS / checkfirst).
    """
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        metadata.create_all(conn)
