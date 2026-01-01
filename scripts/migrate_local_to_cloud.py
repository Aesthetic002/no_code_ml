"""
Migration script to move local data to cloud storage
Run: python scripts/migrate_local_to_cloud.py
"""

import os
import sys
import json
from pathlib import Path

# Add app to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.cloud_storage import CloudStorageConfig, CloudStorageManager
import joblib

def migrate_models_to_cloud():
    """Migrate all local models to cloud storage"""
    
    config = CloudStorageConfig()
    manager = CloudStorageManager(config)
    
    # Local paths
    project_dir = Path(__file__).parent.parent
    models_dir = project_dir / "models"
    
    if not models_dir.exists():
        print("❌ No local models directory found")
        return
    
    print(f"🔄 Starting migration from {models_dir}")
    
    migrated_count = 0
    error_count = 0
    
    # Iterate through users and models
    for username_dir in models_dir.iterdir():
        if not username_dir.is_dir():
            continue
        
        username = username_dir.name
        print(f"\n👤 Processing user: {username}")
        
        # Process regression and classification models
        for model_type_dir in username_dir.iterdir():
            if not model_type_dir.is_dir():
                continue
            
            model_type = model_type_dir.name
            
            for model_file in model_type_dir.glob("*.pkl"):
                try:
                    model_name = model_file.stem
                    
                    # Read model
                    with open(model_file, 'rb') as f:
                        model_data = f.read()
                    
                    # Upload to cloud
                    remote_path = manager.upload_model(
                        model_data, 
                        username, 
                        model_name,
                        model_type
                    )
                    
                    print(f"  ✅ {model_name}.pkl → {remote_path}")
                    migrated_count += 1
                    
                except Exception as e:
                    print(f"  ❌ Error uploading {model_file.name}: {str(e)}")
                    error_count += 1
    
    print(f"\n📊 Migration Complete!")
    print(f"✅ Successfully uploaded: {migrated_count} models")
    print(f"❌ Errors: {error_count}")

if __name__ == "__main__":
    print("=" * 50)
    print("Cloud Storage Migration Tool")
    print("=" * 50)
    
    # Check if .env exists
    env_file = Path(__file__).parent.parent / ".env"
    if not env_file.exists():
        print("\n⚠️  WARNING: .env file not found!")
        print("   Create .env file from .env.example first")
        print("\n   Example:")
        print("   cp .env.example .env")
        print("   # Edit .env with your cloud credentials")
        sys.exit(1)
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    backend = os.getenv("STORAGE_BACKEND", "local")
    print(f"\n📦 Storage Backend: {backend}")
    
    if backend == "local":
        print("⚠️  Storage backend is 'local'. No cloud upload will occur.")
        sys.exit(0)
    
    # Run migration
    migrate_models_to_cloud()
