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
import sqlite3
from datetime import datetime, timedelta
import jwt
from pydantic import BaseModel
import hashlib
import secrets

# Database setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
DB_PATH = os.path.join(PROJECT_DIR, "users.db")

# JWT and Security
SECRET_KEY = "your-secret-key-change-this-in-production"
ALGORITHM = "HS256"
security = HTTPBearer()

# Simple password hashing using PBKDF2
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
    StaticFiles(directory=os.path.join(PROJECT_DIR, "frontend")),
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

# Database initialization
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        hashed_password TEXT NOT NULL,
        is_admin BOOLEAN DEFAULT 0,
        created_at TEXT NOT NULL,
        last_login TEXT
    )''')
    
    # Models metadata table (for admin dashboard)
    c.execute('''CREATE TABLE IF NOT EXISTS models_metadata (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        model_name TEXT NOT NULL,
        model_type TEXT NOT NULL,
        features TEXT NOT NULL,
        accuracy REAL,
        r2_score REAL,
        target_column TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )''')
    
    conn.commit()
    conn.close()

init_db()

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

def get_user_id(username: str) -> int:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    if result:
        return result[0]
    raise HTTPException(status_code=404, detail="User not found")

def get_user_dir(username: str) -> str:
    user_models_dir = os.path.join(MODELS_DIR, username)
    os.makedirs(user_models_dir, exist_ok=True)
    return user_models_dir

# ============ AUTHENTICATION ENDPOINTS ============

@app.post("/api/register", response_model=Token)
async def register(user: UserRegister):
    """Register a new user"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    try:
        hashed_password = hash_password(user.password)
        created_at = datetime.utcnow().isoformat()
        
        c.execute(
            "INSERT INTO users (username, email, hashed_password, created_at) VALUES (?, ?, ?, ?)",
            (user.username, user.email, hashed_password, created_at)
        )
        conn.commit()
        
        access_token = create_access_token(user.username)
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "username": user.username
        }
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Username or email already exists")
    finally:
        conn.close()

@app.post("/api/login", response_model=Token)
async def login(user: UserLogin):
    """Login user"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute("SELECT hashed_password FROM users WHERE username = ?", (user.username,))
    result = c.fetchone()
    conn.close()
    
    if not result or not verify_password(user.password, result[0]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Update last login
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE users SET last_login = ? WHERE username = ?", 
              (datetime.utcnow().isoformat(), user.username))
    conn.commit()
    conn.close()
    
    access_token = create_access_token(user.username)
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "username": user.username
    }

@app.get("/api/user/profile")
async def get_profile(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user profile"""
    username = verify_token(credentials)
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, username, email, is_admin, created_at FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    
    if not result:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        "id": result[0],
        "username": result[1],
        "email": result[2],
        "is_admin": result[3],
        "created_at": result[4]
    }


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
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    username = verify_token(credentials)
    user_id = get_user_id(username)
    
    df = pd.read_csv(file.file)

    target_column = df.columns[-1]
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

    # User-specific directory
    user_dir = get_user_dir(username)
    regression_dir = os.path.join(user_dir, "regression")
    os.makedirs(regression_dir, exist_ok=True)
    
    model_path = os.path.join(regression_dir, f"{model_name}.pkl")
    metadata_path = os.path.join(regression_dir, f"{model_name}_metadata.json")
    
    # Save model
    joblib.dump(model, model_path)
    
    # Save metadata
    metadata = {
        "model_name": model_name,
        "target_column": target_column,
        "features": X.columns.tolist(),
        "model_type": "regression"
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)
    
    # Save to database
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO models_metadata (user_id, model_name, model_type, features, r2_score, target_column, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (user_id, model_name, "regression", ",".join(X.columns.tolist()), score, target_column, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()

    return {
        "model_name": model_name,
        "target_column": target_column,
        "r2_score": round(score, 3),
        "model_saved_as": model_path,
        "features": X.columns.tolist()
    }


@app.post("/predict-regression")
async def predict_regression(model_name: str, data: dict, credentials: HTTPAuthorizationCredentials = Depends(security)):
    username = verify_token(credentials)
    user_dir = get_user_dir(username)
    
    model_path = os.path.join(user_dir, "regression", f"{model_name}.pkl")

    if not os.path.exists(model_path):
        return {"error": "Model not found"}

    model = joblib.load(model_path)
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
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    username = verify_token(credentials)
    user_id = get_user_id(username)
    
    df = pd.read_csv(file.file)

    target_column = df.columns[-1]
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

    # User-specific directory
    user_dir = get_user_dir(username)
    classification_dir = os.path.join(user_dir, "classification")
    os.makedirs(classification_dir, exist_ok=True)
    
    model_path = os.path.join(classification_dir, f"{model_name}.pkl")
    metadata_path = os.path.join(classification_dir, f"{model_name}_metadata.json")
    
    # Save model bundle
    joblib.dump(
        {"model": model, "labels": label_map},
        model_path
    )
    
    # Save metadata
    metadata = {
        "model_name": model_name,
        "target_column": target_column,
        "features": X.columns.tolist(),
        "model_type": "classification",
        "labels": list(label_map.values())
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)
    
    # Save to database
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO models_metadata (user_id, model_name, model_type, features, accuracy, target_column, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (user_id, model_name, "classification", ",".join(X.columns.tolist()), acc, target_column, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()

    return {
        "model_name": model_name,
        "target_column": target_column,
        "accuracy": round(acc, 3),
        "labels": label_map,
        "model_saved_as": model_path,
        "features": X.columns.tolist()
    }

@app.post("/predict-classification")
async def predict_classification(model_name: str, data: dict, credentials: HTTPAuthorizationCredentials = Depends(security)):
    username = verify_token(credentials)
    user_dir = get_user_dir(username)
    
    model_path = os.path.join(user_dir, "classification", f"{model_name}.pkl")

    if not os.path.exists(model_path):
        return {"error": "Model not found"}

    bundle = joblib.load(model_path)
    model = bundle["model"]
    label_map = bundle["labels"]

    input_df = pd.DataFrame([data])
    pred_code = model.predict(input_df)[0]
    pred_label = label_map[pred_code]

    return {
        "model_used": model_name,
        "predicted_class": pred_label
    }


@app.get("/models/{model_type}")
def list_models(model_type: str, credentials: HTTPAuthorizationCredentials = Depends(security)):
    username = verify_token(credentials)
    user_dir = get_user_dir(username)
    base_path = os.path.join(user_dir, model_type)

    if not os.path.exists(base_path):
        return {"available_models": []}

    models = [
        file.replace(".pkl", "")
        for file in os.listdir(base_path)
        if file.endswith(".pkl")
    ]

    return {
        "model_type": model_type,
        "available_models": models
    }


@app.get("/model-metadata/{model_type}/{model_name}")
def get_model_metadata(model_type: str, model_name: str, credentials: HTTPAuthorizationCredentials = Depends(security)):
    username = verify_token(credentials)
    user_dir = get_user_dir(username)
    
    metadata_path = os.path.join(user_dir, model_type, f"{model_name}_metadata.json")
    
    if not os.path.exists(metadata_path):
        return {"error": "Model metadata not found"}
    
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    return metadata


# ============ ADMIN ENDPOINTS ============

def verify_admin(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify user is admin"""
    username = verify_token(credentials)
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT is_admin FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    
    if not result or not result[0]:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    return username

@app.get("/api/admin/dashboard")
async def admin_dashboard(admin_user: str = Depends(verify_admin)):
    """Get admin dashboard stats"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Get user stats
    c.execute("SELECT COUNT(*) FROM users")
    total_users = c.fetchone()[0]
    
    c.execute("SELECT COUNT(*) FROM models_metadata")
    total_models = c.fetchone()[0]
    
    # Get recent models
    c.execute("""
        SELECT u.username, m.model_name, m.model_type, m.created_at 
        FROM models_metadata m 
        JOIN users u ON m.user_id = u.id 
        ORDER BY m.created_at DESC 
        LIMIT 10
    """)
    recent_models = c.fetchall()
    
    conn.close()
    
    return {
        "total_users": total_users,
        "total_models": total_models,
        "recent_models": [
            {
                "username": m[0],
                "model_name": m[1],
                "model_type": m[2],
                "created_at": m[3]
            } for m in recent_models
        ]
    }

@app.get("/api/admin/users")
async def get_all_users(admin_user: str = Depends(verify_admin)):
    """Get all users list"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute("""
        SELECT id, username, email, is_admin, created_at, last_login 
        FROM users 
        ORDER BY created_at DESC
    """)
    users = c.fetchall()
    conn.close()
    
    return {
        "users": [
            {
                "id": u[0],
                "username": u[1],
                "email": u[2],
                "is_admin": u[3],
                "created_at": u[4],
                "last_login": u[5]
            } for u in users
        ]
    }

@app.post("/api/admin/users/{user_id}/toggle-admin")
async def toggle_admin(user_id: int, admin_user: str = Depends(verify_admin)):
    """Toggle user admin status"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute("SELECT is_admin FROM users WHERE id = ?", (user_id,))
    result = c.fetchone()
    
    if not result:
        raise HTTPException(status_code=404, detail="User not found")
    
    new_status = not result[0]
    c.execute("UPDATE users SET is_admin = ? WHERE id = ?", (new_status, user_id))
    conn.commit()
    conn.close()
    
    return {"success": True, "is_admin": new_status}

@app.delete("/api/admin/users/{user_id}")
async def delete_user(user_id: int, admin_user: str = Depends(verify_admin)):
    """Delete a user and their models"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute("SELECT username FROM users WHERE id = ?", (user_id,))
    result = c.fetchone()
    
    if not result:
        raise HTTPException(status_code=404, detail="User not found")
    
    username = result[0]
    
    # Delete models metadata
    c.execute("DELETE FROM models_metadata WHERE user_id = ?", (user_id,))
    
    # Delete user
    c.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()
    
    # Delete user directory
    user_dir = os.path.join(MODELS_DIR, username)
    if os.path.exists(user_dir):
        import shutil
        shutil.rmtree(user_dir)
    
    return {"success": True, "message": f"User {username} deleted"}

@app.get("/api/admin/user-models/{user_id}")
async def get_user_models(user_id: int, admin_user: str = Depends(verify_admin)):
    """Get all models of a specific user"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute("SELECT username FROM users WHERE id = ?", (user_id,))
    user_result = c.fetchone()
    
    if not user_result:
        raise HTTPException(status_code=404, detail="User not found")
    
    c.execute("""
        SELECT model_name, model_type, accuracy, r2_score, created_at 
        FROM models_metadata 
        WHERE user_id = ? 
        ORDER BY created_at DESC
    """, (user_id,))
    
    models = c.fetchall()
    conn.close()
    
    return {
        "username": user_result[0],
        "models": [
            {
                "model_name": m[0],
                "model_type": m[1],
                "accuracy": m[2],
                "r2_score": m[3],
                "created_at": m[4]
            } for m in models
        ]
    }
