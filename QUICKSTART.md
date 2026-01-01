# Quick Start: Cloud Storage Setup

## For Complete Beginners - Use Supabase (Easiest)

### 1. Create Supabase Account (5 minutes)
```
1. Go to https://supabase.com
2. Click "Sign Up" → Sign with GitHub
3. Create Organization → Create Project
4. Wait 30 seconds for setup
```

### 2. Get Your Credentials (2 minutes)
In Supabase Dashboard:
```
1. Settings → Database
   Copy the CONNECTION STRING (looks like: postgresql://postgres:xxx@xxx.supabase.co:5432/postgres)

2. Settings → API  
   Copy PROJECT URL (looks like: https://xxx.supabase.co)
   Copy ANON KEY (long string starting with eyJ)

3. Storage → Create Bucket
   Name: models
   Make PUBLIC
```

### 3. Create .env File (1 minute)
In your project root folder, create file `.env`:
```
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@xxx.supabase.co:5432/postgres
STORAGE_BACKEND=supabase
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=your_anon_key_here
SECRET_KEY=mysupersecretkey12345678901234567890
```

### 4. Install Packages (2 minutes)
```bash
pip install -r requirements.txt
```

### 5. Run Server (1 minute)
```bash
uvicorn app.main:app --reload --port 9000
```

✅ Done! Your data now syncs to cloud automatically.

---

## For Production - Use AWS (Best Performance)

### Cost
- ~$20/month for database (AWS RDS)
- ~$1-5/month for storage (AWS S3)
- Total: ~$25/month

### Setup (15 minutes)

#### Step 1: Create S3 Bucket
```
1. AWS Console → S3
2. Create Bucket → Name: nocode-ml-models
3. Note the bucket name
```

#### Step 2: Create RDS Database
```
1. AWS Console → RDS
2. Create Database → PostgreSQL
3. Instance name: nocode-ml-db
4. Master username: admin
5. Master password: [generate strong password]
6. Storage: 20GB
7. Wait for "Available" status
8. Copy Endpoint: xxx.rds.amazonaws.com
```

#### Step 3: Create IAM User
```
1. AWS Console → IAM
2. Users → Create User → nocode-ml-app
3. Attach Policy: AmazonS3FullAccess
4. Create Access Key
5. Copy Access Key ID and Secret Access Key
```

#### Step 4: Create .env
```
DATABASE_URL=postgresql://admin:YOUR_PASSWORD@nocode-ml-db.xxxxx.rds.amazonaws.com:5432/postgres
STORAGE_BACKEND=aws_s3
AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_secret_access_key
AWS_REGION=us-east-1
S3_BUCKET_NAME=nocode-ml-models
SECRET_KEY=mysupersecretkey12345678901234567890
```

#### Step 5: Install & Run
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 9000
```

---

## Verify It Works

### Test 1: Check Backend
```bash
# In PowerShell
curl http://127.0.0.1:9000/api/test
# Should return: {"status":"ok","message":"API is working"}
```

### Test 2: Register User
```bash
curl -X POST http://127.0.0.1:9000/api/register `
  -H "Content-Type: application/json" `
  -d '{
    "username": "testuser",
    "email": "test@example.com", 
    "password": "password123"
  }'
```

### Test 3: Train Model
- Go to http://127.0.0.1:9000/frontend
- Upload CSV file
- Train model
- Check Supabase/AWS to confirm upload ✅

---

## Still Using Local Storage?

No problem! Just keep:
```
STORAGE_BACKEND=local
DATABASE_URL=sqlite:///./users.db
```

Models will be saved to `./models/` folder (same as before).

---

## Need Help?

### Error: "Connection refused"
→ Check DATABASE_URL is correct
→ For local: use `sqlite:///./users.db`

### Error: "Invalid credentials"  
→ Check AWS keys/Supabase key
→ Verify S3 bucket name

### Error: "psycopg2" not found
→ Run: `pip install -r requirements.txt`

### Models not uploading
→ Check STORAGE_BACKEND value
→ Verify cloud credentials
→ Check bucket/container exists

---

## File Structure
```
project/
├── .env                    ← Your cloud credentials (KEEP SECRET!)
├── .env.example            ← Template (safe to share)
├── requirements.txt        ← Install: pip install -r
├── CLOUD_SETUP.md          ← Detailed guide
├── INTEGRATION_GUIDE.md    ← How to integrate with main.py
├── app/
│   ├── main.py            ← Update this file
│   ├── cloud_storage.py   ← New: Cloud storage logic
│   └── database.py        ← New: Database models
├── scripts/
│   └── migrate_local_to_cloud.py  ← New: Migration tool
└── frontend/              ← UI files (no changes)
```
