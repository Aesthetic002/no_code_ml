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
from app.langchain_utils import (
    intelligent_eda_analysis,
    generate_data_quality_report,
    recommend_models,
    chat_with_assistant,
    generate_model_report,
    explain_prediction
)

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

def clean_and_validate_data(df: pd.DataFrame, target_column: str = None):
    """
    Comprehensive EDA and data cleaning function
    
    Returns:
        - df_cleaned: Cleaned dataframe ready for ML
        - eda_report: Dictionary with cleaning details
    """
    eda_report = {
        "original_rows": len(df),
        "original_columns": len(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "actions_taken": []
    }
    
    # Step 1: Get numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if not numeric_cols:
        raise HTTPException(
            status_code=400,
            detail="No numeric columns found. Please ensure your dataset has numeric features."
        )
    
    # Step 2: Drop rows where target column is NaN (if specified)
    if target_column and target_column in df.columns:
        initial_rows = len(df)
        df = df.dropna(subset=[target_column])
        rows_dropped = initial_rows - len(df)
        if rows_dropped > 0:
            eda_report["actions_taken"].append(f"Dropped {rows_dropped} rows with missing target value")
    
    # Step 3: Select only numeric columns
    df_numeric = df[numeric_cols].copy()
    eda_report["numeric_features"] = numeric_cols
    
    # Step 4: Handle missing values in features
    # Strategy: Drop rows with ANY missing values (strict approach for clean training)
    initial_rows = len(df_numeric)
    df_numeric = df_numeric.dropna()
    rows_dropped = initial_rows - len(df_numeric)
    
    if rows_dropped > 0:
        eda_report["actions_taken"].append(
            f"Dropped {rows_dropped} rows with missing values in features ({(rows_dropped/initial_rows*100):.1f}%)"
        )
    
    # Step 5: Validate we have enough data
    if len(df_numeric) < 10:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough data after cleaning. Have {len(df_numeric)} rows, need at least 10. "
                   f"Your dataset may have too many missing values."
        )
    
    # Step 6: Check for infinite values
    if (df_numeric == float('inf')).any().any() or (df_numeric == float('-inf')).any().any():
        df_numeric = df_numeric.replace([float('inf'), float('-inf')], float('nan')).dropna()
        eda_report["actions_taken"].append("Removed rows with infinite values")
    
    # Step 7: Summary statistics
    eda_report["cleaned_rows"] = len(df_numeric)
    eda_report["rows_retained_percent"] = (len(df_numeric) / eda_report["original_rows"] * 100) if eda_report["original_rows"] > 0 else 0
    eda_report["final_features"] = len(df_numeric.columns)
    eda_report["data_quality"] = {
        "missing_before": sum(1 for col in eda_report["missing_values"].values() if col > 0),
        "missing_after": sum(1 for col in df_numeric.isnull().sum().to_dict().values() if col > 0),
    }
    
    print(f"📊 EDA Report:")
    print(f"   Original: {eda_report['original_rows']} rows, {eda_report['original_columns']} columns")
    print(f"   Cleaned: {eda_report['cleaned_rows']} rows, {eda_report['final_features']} features")
    print(f"   Retained: {eda_report['rows_retained_percent']:.1f}% of data")
    for action in eda_report["actions_taken"]:
        print(f"   ✓ {action}")
    
    return df_numeric, eda_report

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
    
    # Clean and validate data
    try:
        df_cleaned, eda_report = clean_and_validate_data(df, target_column)
        feature_columns = df_cleaned.columns.tolist()
        if target_column in feature_columns:
            feature_columns.remove(target_column)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Data validation failed: {str(e)}")

    return {
        "rows": eda_report["original_rows"],
        "cleaned_rows": eda_report["cleaned_rows"],
        "columns": eda_report["original_columns"],
        "target_column": target_column,
        "feature_columns": feature_columns,
        "data_quality": eda_report["data_quality"],
        "cleaning_actions": eda_report["actions_taken"],
        "rows_retained_percent": round(eda_report["rows_retained_percent"], 2)
    }

@app.post("/ai-analyze")
async def ai_analyze_csv(file: UploadFile = File(...), credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Feature #1: AI-powered intelligent data analysis
    Uses Gemini to provide insights about the dataset
    """
    username = verify_token(credentials)
    df = pd.read_csv(file.file)
    target_column = df.columns[-1]
    
    try:
        # Get AI insights
        ai_results = intelligent_eda_analysis(df, target_column)
        
        if not ai_results.get("success"):
            raise HTTPException(status_code=500, detail="AI analysis failed")
        
        analysis = ai_results["analysis"]
        
        # Get data quality report
        quality_report = generate_data_quality_report(df)
        
        return {
            "success": True,
            "problem_type": analysis.get("problem_type", "Unknown"),
            "data_quality_score": analysis.get("data_quality_score", "N/A"),
            "key_insights": analysis.get("key_insights", []),
            "warning_flags": analysis.get("warning_flags", []),
            "recommended_features": analysis.get("recommended_features", []),
            "next_steps": analysis.get("next_steps", []),
            "quality_report": quality_report,
            "target_column": target_column
        }
    except Exception as e:
        print(f"❌ AI Analysis Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AI analysis failed: {str(e)}")

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
    
    # Clean and validate data
    try:
        df_cleaned, eda_report = clean_and_validate_data(df, target_column)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Data cleaning failed: {str(e)}")
    
    # Separate features and target from cleaned data
    X = df_cleaned.drop(columns=[target_column], errors='ignore')
    y = df_cleaned[target_column]

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
    
    # Clean and validate data
    try:
        df_cleaned, eda_report = clean_and_validate_data(df, target_column)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Data cleaning failed: {str(e)}")
    
    # Separate features and target from cleaned data
    y = df_cleaned[target_column].astype("category")
    label_map = dict(enumerate(y.cat.categories))
    
    X = df_cleaned.drop(columns=[target_column], errors='ignore')
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

@app.get("/api/admin/all-models")
async def get_all_models(
    admin_user: str = Depends(verify_admin),
    db: Session = Depends(get_db)
):
    """Get all models from all users with user information"""
    
    models = db.query(ModelMetadata).order_by(ModelMetadata.created_at.desc()).all()
    
    models_list = []
    for m in models:
        user = db.query(User).filter(User.id == m.user_id).first()
        models_list.append({
            "id": m.id,
            "model_name": m.model_name,
            "model_type": m.model_type,
            "username": user.username if user else "Unknown",
            "user_id": m.user_id,
            "features": m.features,
            "accuracy": m.accuracy,
            "r2_score": m.r2_score,
            "target_column": m.target_column,
            "created_at": m.created_at.isoformat() if m.created_at else None
        })
    
    return {
        "models": models_list
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
