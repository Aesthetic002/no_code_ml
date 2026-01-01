import os
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, Float, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Determine database backend
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./users.db"  # Fallback to SQLite
)

print(f"📊 Configured database: {DATABASE_URL[:50]}...")

# For SQLite
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    # For PostgreSQL (cloud)
    try:
        engine = create_engine(DATABASE_URL, pool_pre_ping=True, echo=False)
        # Test connection
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        print("✅ PostgreSQL connection successful")
    except Exception as e:
        print(f"⚠️  PostgreSQL connection failed: {e}")
        print("   Falling back to SQLite...")
        # Fallback to SQLite
        DATABASE_URL = "sqlite:///./users.db"
        engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)

class ModelMetadata(Base):
    __tablename__ = "models_metadata"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True, nullable=False)
    model_name = Column(String, nullable=False)
    model_type = Column(String, nullable=False)  # 'regression' or 'classification'
    features = Column(Text, nullable=False)  # JSON string of feature list
    accuracy = Column(Float, nullable=True)
    r2_score = Column(Float, nullable=True)
    target_column = Column(String, nullable=False)
    cloud_path = Column(String, nullable=True)  # S3 or Supabase URL
    created_at = Column(DateTime, default=datetime.utcnow)

# Create tables
try:
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created successfully")
except Exception as e:
    print(f"⚠️  Error creating tables: {e}")

def get_db():
    """Dependency for getting database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
