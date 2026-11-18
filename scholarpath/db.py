"""Database layer (SQLite by default, PostgreSQL when DATABASE_URL is set).

Everything the system must remember lives here, including the uploaded
datasets and the saved Joblib model files, so a deployment only has one
thing to keep safe.
"""
import json
from datetime import datetime, timezone

from sqlalchemy import (Boolean, DateTime, Float, ForeignKey, Integer,
                        LargeBinary, String, Text, create_engine)
from sqlalchemy.orm import (DeclarativeBase, Mapped, mapped_column,
                            relationship, scoped_session, sessionmaker)


def now():
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    email: Mapped[str] = mapped_column(String(180), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(20), default="user")   # user | admin
    occupation: Mapped[str] = mapped_column(String(40), default="Student")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)

    @property
    def is_admin(self):
        return self.role == "admin"


class Dataset(Base):
    __tablename__ = "datasets"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(160))
    note: Mapped[str] = mapped_column(Text, default="")
    simulated: Mapped[bool] = mapped_column(Boolean, default=False)
    csv: Mapped[bytes] = mapped_column(LargeBinary)
    rows_raw: Mapped[int] = mapped_column(Integer)
    rows_clean: Mapped[int] = mapped_column(Integer)
    report_json: Mapped[str] = mapped_column(Text)
    uploaded_by: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)
    models = relationship("ModelVersion", back_populates="dataset")

    @property
    def report(self):
        return json.loads(self.report_json)


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    version: Mapped[int] = mapped_column(Integer)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"))
    params_json: Mapped[str] = mapped_column(Text)
    report_json: Mapped[str] = mapped_column(Text)
    accuracy: Mapped[float] = mapped_column(Float)
    macro_f1: Mapped[float] = mapped_column(Float)
    blob: Mapped[bytes] = mapped_column(LargeBinary)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    trained_by: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)
    dataset = relationship("Dataset", back_populates="models")

    @property
    def params(self):
        return json.loads(self.params_json)

    @property
    def report(self):
        return json.loads(self.report_json)


class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"))
    reference: Mapped[str] = mapped_column(String(120), default="")
    institution_type: Mapped[str] = mapped_column(String(40))
    inputs_json: Mapped[str] = mapped_column(Text)
    tier: Mapped[int] = mapped_column(Integer)
    label: Mapped[str] = mapped_column(String(60))
    confidence: Mapped[float] = mapped_column(Float)
    result_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)
    user = relationship("User")
    model = relationship("ModelVersion")

    @property
    def inputs(self):
        return json.loads(self.inputs_json)

    @property
    def result(self):
        return json.loads(self.result_json)


def make_session(url):
    kwargs = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    engine = create_engine(url, **kwargs)
    Base.metadata.create_all(engine)
    return scoped_session(sessionmaker(bind=engine, expire_on_commit=False)), engine
