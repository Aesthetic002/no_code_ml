from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib
import os
from typing import Dict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import json
import io
from datetime import datetime, timedelta
import jwt
from pydantic import BaseModel
import hashlib
import secrets
from dotenv import load_dotenv
from sqlalchemy.orm import Session

# Load environment variables
load_dotenv()

# Import cloud storage and database
from app.cloud_storage import CloudStorageConfig, CloudStorageManager
from app.database import get_db, User, ModelMetadata

# Database setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

# JWT and Security
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-this-in-production")
ALGORITHM = "HS256"
security = HTTPBearer()

# Cloud Storage Setup
storage_config = CloudStorageConfig()
storage_manager = CloudStorageManager(storage_config)
backend = os.getenv("STORAGE_BACKEND", "local")
print(f"🚀 Using {backend} storage backend")

# Helper functions
def hash_password(password: str) -> str:
    salt = secrets.token_hex(32)
    pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
    return f"{salt}${pwd_hash.hex()}"

def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        salt, pwd_hash = hashed_password.split('$')
        new_hash = hashlib.pbkdf2_hmac('sha256', plain_password.encode(), salt.encode(), 100000)
        return new_hash.hex() == pwd_hash
    except:
        return False

def create_access_token(username: str, expires_delta: timedelta = None):
    if expires_delta is None:
        expires_delta = timedelta(days=7)
    
    expire = datetime.utcnow() + expires_delta
    to_encode = {"sub": username, "exp": expire}
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials) -> str:
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def get_user_id_from_db(username: str, db: Session) -> int:
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user.id

def get_user_dir(username: str) -> str:
    """Keep this for local storage fallback"""
    user_models_dir = os.path.join(MODELS_DIR, username)
    os.makedirs(user_models_dir, exist_ok=True)
    return user_models_dir

# Pydantic models
class UserRegister(BaseModel):
    username: str
    email: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    username: str

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
app.mount(
    "/frontend",
    StaticFiles(directory=os.path.join(PROJECT_DIR, "frontend"), html=True),
    name="frontend"
)

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    index_path = os.path.join(PROJECT_DIR, "frontend", "index.html")
    with open(index_path) as f:
        return f.read()

@app.get("/api/test")
def test_endpoint():
    return {"status": "ok", "message": "API is working"}

# ============ AUTHENTICATION ENDPOINTS ============

@app.post("/api/register", response_model=Token)
async def register(user: UserRegister, db: Session = Depends(get_db)):
    """Register a new user"""
    try:
        existing_user = db.query(User).filter(
            (User.username == user.username) | (User.email == user.email)
        ).first()
        
        if existing_user:
            raise HTTPException(status_code=400, detail="Username or email already exists")
        
        hashed_password = hash_password(user.password)
        new_user = User(
            username=user.username,
            email=user.email,
            hashed_password=hashed_password
        )
        db.add(new_user)
        db.commit()
        
        access_token = create_access_token(user.username)
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "username": user.username
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/login", response_model=Token)
async def login(user: UserLogin, db: Session = Depends(get_db)):
    """Login user"""
    db_user = db.query(User).filter(User.username == user.username).first()
    
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Update last login
    db_user.last_login = datetime.utcnow()
    db.commit()
    
    access_token = create_access_token(user.username)
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "username": user.username
    }

@app.get("/api/user/profile")
async def get_profile(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    """Get current user profile"""
    username = verify_token(credentials)
    
    user = db.query(User).filter(User.username == username).first()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "is_admin": user.is_admin,
        "created_at": user.created_at.isoformat() if user.created_at else None
    }

# ============ ML ENDPOINTS ============

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)

    return {
        "filename": file.filename,
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": df.columns.tolist()
    }

@app.post("/analyze")
async def analyze_csv(file: UploadFile = File(...), credentials: HTTPAuthorizationCredentials = Depends(security)):
    username = verify_token(credentials)
    df = pd.read_csv(file.file)

    target_column = df.columns[-1]
    feature_columns = df.drop(columns=[target_column]).select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "target_column": target_column,
        "feature_columns": feature_columns
    }

@app.post("/detect-problem")
async def detect_problem(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)

    target_column = df.columns[-1]
    target_dtype = df[target_column].dtype

    if target_dtype == 'object':
        problem_type = "Classification"
    else:
        problem_type = "Regression"

    return {
        "target_column": target_column,
        "data_type": str(target_dtype),
        "problem_type": problem_type
    }

@app.post("/train-regression")
async def train_regression(
    model_name: str,
    file: UploadFile = File(...),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    username = verify_token(credentials)
    user_id = get_user_id_from_db(username, db)
    
    df = pd.read_csv(file.file)

    target_column = df.columns[-1]
    
    # Validate target column is numeric for regression
    if not pd.api.types.is_numeric_dtype(df[target_column]):
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_column}' must be numeric for regression. "
                   f"Found type: {df[target_column].dtype}. "
                   f"Use classification for categorical targets."
        )
    
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X = X.select_dtypes(include=['int64', 'float64'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)

    # Save model to cloud
    model_bytes = io.BytesIO()
    joblib.dump(model, model_bytes)
    model_bytes.seek(0)  # Reset position to start
    model_data = model_bytes.getvalue()
    
    try:
        cloud_path = storage_manager.upload_model(
            model_data,
            username,
            model_name,
            "regression"
        )
        print(f"✅ Model uploaded to: {cloud_path}")
    except Exception as e:
        print(f"❌ Upload failed: {str(e)}")
        return {"error": f"Failed to upload model: {str(e)}"}
    
    # Save metadata to database
    db.add(ModelMetadata(
        user_id=user_id,
        model_name=model_name,
        model_type="regression",
        features=",".join(X.columns.tolist()),
        r2_score=float(score),
        target_column=target_column,
        cloud_path=cloud_path
    ))
    db.commit()

    return {
        "model_name": model_name,
        "target_column": target_column,
        "r2_score": round(score, 3),
        "model_saved_as": cloud_path,
        "features": X.columns.tolist()
    }

@app.post("/predict-regression")
async def predict_regression(
    model_name: str,
    data: dict,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    username = verify_token(credentials)
    user_id = get_user_id_from_db(username, db)
    
    # Get model from database
    db_model = db.query(ModelMetadata).filter(
        (ModelMetadata.model_name == model_name) &
        (ModelMetadata.user_id == user_id) &
        (ModelMetadata.model_type == "regression")
    ).first()
    
    if not db_model:
        return {"error": "Model not found"}
    
    # Download from cloud
    try:
        model_data = storage_manager.download_model(db_model.cloud_path)
        model = joblib.load(io.BytesIO(model_data))
    except Exception as e:
        return {"error": f"Failed to load model: {str(e)}"}
    
    input_df = pd.DataFrame([data])
    prediction = model.predict(input_df)[0]

    return {
        "model_used": model_name,
        "prediction": round(float(prediction), 2)
    }

@app.post("/train-classification")
async def train_classification(
    model_name: str,
    file: UploadFile = File(...),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    username = verify_token(credentials)
    user_id = get_user_id_from_db(username, db)
    
    df = pd.read_csv(file.file)

    target_column = df.columns[-1]
    
    # Validate target column is categorical for classification
    if pd.api.types.is_numeric_dtype(df[target_column]):
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_column}' should be categorical for classification. "
                   f"Found numeric type: {df[target_column].dtype}. "
                   f"Use regression for numeric targets."
        )
    
    y = df[target_column].astype("category")

    label_map = dict(enumerate(y.cat.categories))

    X = df.drop(columns=[target_column])
    X = X.select_dtypes(include=['int64', 'float64'])
    y_encoded = y.cat.codes

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))

    # Save model bundle to cloud
    model_bytes = io.BytesIO()
    joblib.dump(
        {"model": model, "labels": label_map},
        model_bytes
    )
    model_bytes.seek(0)  # Reset position to start
    model_data = model_bytes.getvalue()
    
    try:
        cloud_path = storage_manager.upload_model(
            model_data,
            username,
            model_name,
            "classification"
        )
        print(f"✅ Model uploaded to: {cloud_path}")
    except Exception as e:
        print(f"❌ Upload failed: {str(e)}")
        return {"error": f"Failed to upload model: {str(e)}"}
    
    # Save metadata to database
    db.add(ModelMetadata(
        user_id=user_id,
        model_name=model_name,
        model_type="classification",
        features=",".join(X.columns.tolist()),
        accuracy=float(acc),
        target_column=target_column,
        cloud_path=cloud_path
    ))
    db.commit()

    return {
        "model_name": model_name,
        "target_column": target_column,
        "accuracy": round(acc, 3),
        "labels": label_map,
        "model_saved_as": cloud_path,
        "features": X.columns.tolist()
    }

@app.post("/predict-classification")
async def predict_classification(
    model_name: str,
    data: dict,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    username = verify_token(credentials)
    user_id = get_user_id_from_db(username, db)
    
    # Get model from database
    db_model = db.query(ModelMetadata).filter(
        (ModelMetadata.model_name == model_name) &
        (ModelMetadata.user_id == user_id) &
        (ModelMetadata.model_type == "classification")
    ).first()
    
    if not db_model:
        return {"error": "Model not found"}
    
    # Download from cloud
    try:
        model_data = storage_manager.download_model(db_model.cloud_path)
        bundle = joblib.load(io.BytesIO(model_data))
        model = bundle["model"]
        label_map = bundle["labels"]
    except Exception as e:
        return {"error": f"Failed to load model: {str(e)}"}

    input_df = pd.DataFrame([data])
    pred_code = model.predict(input_df)[0]
    pred_label = label_map[pred_code]

    return {
        "model_used": model_name,
        "predicted_class": pred_label
    }

@app.get("/models/{model_type}")
def list_models(model_type: str, credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    username = verify_token(credentials)
    user_id = get_user_id_from_db(username, db)
    
    models = db.query(ModelMetadata).filter(
        (ModelMetadata.user_id == user_id) &
        (ModelMetadata.model_type == model_type)
    ).all()
    
    return {
        "model_type": model_type,
        "available_models": [m.model_name for m in models]
    }

@app.get("/model-metadata/{model_type}/{model_name}")
def get_model_metadata(
    model_type: str,
    model_name: str,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    username = verify_token(credentials)
    user_id = get_user_id_from_db(username, db)
    
    db_model = db.query(ModelMetadata).filter(
        (ModelMetadata.model_name == model_name) &
        (ModelMetadata.model_type == model_type) &
        (ModelMetadata.user_id == user_id)
    ).first()
    
    if not db_model:
        return {"error": "Model metadata not found"}
    
    return {
        "model_name": db_model.model_name,
        "model_type": db_model.model_type,
        "target_column": db_model.target_column,
        "features": db_model.features.split(","),
        "accuracy": db_model.accuracy,
        "r2_score": db_model.r2_score,
        "created_at": db_model.created_at.isoformat() if db_model.created_at else None
    }

# ============ ADMIN ENDPOINTS ============

def verify_admin(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)) -> str:
    """Verify user is admin"""
    username = verify_token(credentials)
    
    user = db.query(User).filter(User.username == username).first()
    
    if not user or not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    return username

@app.get("/api/admin/dashboard")
async def admin_dashboard(
    admin_user: str = Depends(verify_admin),
    db: Session = Depends(get_db)
):
    """Get admin dashboard stats"""
    
    total_users = db.query(User).count()
    total_models = db.query(ModelMetadata).count()
    
    recent_models = db.query(ModelMetadata).order_by(
        ModelMetadata.created_at.desc()
    ).limit(10).all()
    
    return {
        "total_users": total_users,
        "total_models": total_models,
        "active_users": total_users,
        "recent_models": [
            {
                "username": db.query(User).filter(User.id == m.user_id).first().username,
                "model_name": m.model_name,
                "model_type": m.model_type,
                "created_at": m.created_at.isoformat() if m.created_at else None
            } for m in recent_models
        ]
    }

@app.get("/api/admin/users")
async def get_all_users(
    admin_user: str = Depends(verify_admin),
    db: Session = Depends(get_db)
):
    """Get all users list"""
    
    users = db.query(User).order_by(User.created_at.desc()).all()
    
    return {
        "users": [
            {
                "id": u.id,
                "username": u.username,
                "email": u.email,
                "is_admin": u.is_admin,
                "created_at": u.created_at.isoformat() if u.created_at else None,
                "last_login": u.last_login.isoformat() if u.last_login else None
            } for u in users
        ]
    }

@app.post("/api/admin/users/{user_id}/toggle-admin")
async def toggle_admin(
    user_id: int,
    admin_user: str = Depends(verify_admin),
    db: Session = Depends(get_db)
):
    """Toggle user admin status"""
    
    user = db.query(User).filter(User.id == user_id).first()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    user.is_admin = not user.is_admin
    db.commit()
    
    return {"success": True, "is_admin": user.is_admin}

@app.delete("/api/admin/users/{user_id}")
async def delete_user(
    user_id: int,
    admin_user: str = Depends(verify_admin),
    db: Session = Depends(get_db)
):
    """Delete a user and their models"""
    
    user = db.query(User).filter(User.id == user_id).first()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    username = user.username
    
    # Delete models metadata
    db.query(ModelMetadata).filter(ModelMetadata.user_id == user_id).delete()
    
    # Delete user
    db.delete(user)
    db.commit()
    
    # Delete local user directory if exists
    user_dir = os.path.join(MODELS_DIR, username)
    if os.path.exists(user_dir):
        import shutil
        shutil.rmtree(user_dir)
    
    return {"success": True, "message": f"User {username} deleted"}

@app.get("/api/admin/user-models/{user_id}")
async def get_user_models(
    user_id: int,
    admin_user: str = Depends(verify_admin),
    db: Session = Depends(get_db)
):
    """Get all models of a specific user"""
    
    user = db.query(User).filter(User.id == user_id).first()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    models = db.query(ModelMetadata).filter(
        ModelMetadata.user_id == user_id
    ).order_by(ModelMetadata.created_at.desc()).all()
    
    return {
        "username": user.username,
        "models": [
            {
                "model_name": m.model_name,
                "model_type": m.model_type,
                "accuracy": m.accuracy,
                "r2_score": m.r2_score,
                "created_at": m.created_at.isoformat() if m.created_at else None
            } for m in models
        ]
    }
