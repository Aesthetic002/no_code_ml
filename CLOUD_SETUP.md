# Cloud Storage & Database Setup Guide

## Overview
This guide explains how to migrate from local SQLite + file storage to cloud-based storage solutions.

## Option 1: AWS S3 + AWS RDS PostgreSQL (Production Recommended)

### Step 1: Create AWS S3 Bucket
1. Go to [AWS S3 Console](https://s3.console.aws.amazon.com)
2. Create a new bucket (e.g., `nocode-ml-models`)
3. Note your bucket name

### Step 2: Create AWS RDS PostgreSQL Database
1. Go to [AWS RDS Console](https://console.aws.amazon.com/rds)
2. Create PostgreSQL database
   - Engine: PostgreSQL 14+
   - DB instance identifier: `nocode-ml-db`
   - Master username: `admin`
   - Master password: [Generate strong password]
   - Storage: 20 GB (free tier eligible)
3. Wait for database to be available
4. Note the endpoint (e.g., `nocode-ml-db.xxxxx.rds.amazonaws.com`)

### Step 3: Create IAM User for S3 Access
1. Go to [AWS IAM Console](https://console.aws.amazon.com/iam)
2. Create new user: `nocode-ml-app`
3. Attach policy: `AmazonS3FullAccess`
4. Generate Access Key ID and Secret Access Key
5. Save these credentials

### Step 4: Update .env File
Create `.env` file in project root:
```
DATABASE_URL=postgresql://admin:YourPassword@nocode-ml-db.xxxxx.rds.amazonaws.com:5432/nocode_ml
STORAGE_BACKEND=aws_s3
AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_secret_access_key
AWS_REGION=us-east-1
S3_BUCKET_NAME=nocode-ml-models
SECRET_KEY=your-random-secret-key-32-chars
```

### Step 5: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 6: Run Server
```bash
uvicorn app.main:app --reload --port 9000
```

---

## Option 2: Supabase (Easiest for Beginners)

### Step 1: Create Supabase Project
1. Go to [Supabase](https://supabase.com)
2. Click "New Project"
3. Fill in project details
4. Wait for project initialization

### Step 2: Get Connection String
1. In Supabase, go to Settings > Database
2. Copy the "Connection string" (PostgreSQL)
3. Replace password with your database password

### Step 3: Create Storage Bucket
1. Go to Storage in Supabase dashboard
2. Create new bucket: `models`
3. Make it public

### Step 4: Get API Keys
1. Go to Settings > API
2. Copy `Project URL` and `anon public` key

### Step 5: Update .env File
```
DATABASE_URL=postgresql://postgres:YourPassword@db.xxxxx.supabase.co:5432/postgres
STORAGE_BACKEND=supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_anon_key
SUPABASE_BUCKET=models
SECRET_KEY=your-random-secret-key-32-chars
```

### Step 6: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 7: Run Server
```bash
uvicorn app.main:app --reload --port 9000
```

---

## Option 3: Google Cloud Storage + Cloud SQL

### Step 1: Create GCP Project
1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create new project

### Step 2: Create Cloud SQL PostgreSQL Instance
1. Go to Cloud SQL > Create Instance
2. Choose PostgreSQL
3. Instance ID: `nocode-ml-db`
4. PostgreSQL version: 14+
5. Region: us-central1

### Step 3: Create Storage Bucket
1. Go to Cloud Storage > Create Bucket
2. Bucket name: `nocode-ml-models-[unique-id]`
3. Location: Multi-region (US)

### Step 4: Create Service Account
1. Go to IAM & Admin > Service Accounts
2. Create new account
3. Grant roles:
   - Cloud SQL Client
   - Storage Object Creator
4. Create JSON key file

### Step 5: Update .env File
```
DATABASE_URL=postgresql://postgres:password@[CLOUD-SQL-IP]:5432/nocode_ml
STORAGE_BACKEND=aws_s3  # Use boto3 for GCS (install google-cloud-storage)
S3_BUCKET_NAME=nocode-ml-models-[unique-id]
SECRET_KEY=your-random-secret-key-32-chars
```

---

## Migration from Local Storage

### Step 1: Export Existing Data
```bash
python scripts/migrate_local_to_cloud.py
```

### Step 2: Verify Cloud Storage
- Check AWS S3 bucket or Supabase for uploaded models
- Check database for migrated metadata

### Step 3: Test Functionality
```bash
# Register new user
curl -X POST http://127.0.0.1:9000/api/register \
  -H "Content-Type: application/json" \
  -d '{"username":"testuser","email":"test@example.com","password":"password123"}'

# Train model (should upload to cloud)
# Use the dashboard UI
```

---

## Cost Estimates

| Service | Provider | Monthly Cost |
|---------|----------|--------------|
| Database | AWS RDS | ~$20-40 |
| Object Storage | AWS S3 | ~$5-10 (1000 models) |
| Database | Supabase | ~$10-25 |
| Storage | Supabase | ~$5-10 |
| Database | Google Cloud SQL | ~$30-50 |
| Storage | Google Cloud | ~$5-10 |

---

## Troubleshooting

### Connection Error: "psycopg2 connection refused"
- Check DATABASE_URL is correct
- Verify database security groups allow your IP
- For local testing, use SQLite

### S3 Upload Error: "Access Denied"
- Verify AWS credentials are correct
- Check IAM user has S3 permissions
- Verify bucket name is correct

### Supabase Upload Error
- Check API key is valid
- Verify bucket exists
- Check bucket is public

---

## Local Development (No Cloud)

To keep using local storage during development:
```
STORAGE_BACKEND=local
DATABASE_URL=sqlite:///./users.db
```

All files will be stored in `./models/` directory.
