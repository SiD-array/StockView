# Render Setup Quick Reference

## Start Command (for manual configuration)

If Root Directory is set to `backend`:
```
uvicorn main:app --host 0.0.0.0 --port $PORT
```

If Root Directory is NOT set (project root):
```
cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT
```

## Complete Manual Configuration

If `render.yaml` is not auto-detected, use these settings:

- **Name**: `stockview-backend`
- **Region**: Choose closest to your users
- **Branch**: `main`
- **Root Directory**: `backend` (recommended)
- **Runtime**: `Python 3`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Plan**: `Free`

## Environment Variables

Render automatically sets `$PORT` - no need to configure it manually.

Optional: Set `PYTHON_VERSION=3.12.0` if needed.

