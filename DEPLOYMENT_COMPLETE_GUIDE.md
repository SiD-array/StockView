# 🚀 Complete Deployment Guide - StockView

This guide provides step-by-step instructions for deploying StockView frontend on Vercel and backend on Render.

---

## 📋 Prerequisites Checklist

Before starting, ensure you have:
- [ ] GitHub repository with your code pushed
- [ ] Vercel account (free tier available)
- [ ] Render account (free tier available)
- [ ] Firebase account with Firestore enabled
- [ ] Your backend URL from Render (will get this during deployment)

---

## 🔧 Part 1: Backend Deployment on Render

### Step 1: Create Render Account
1. Go to [render.com](https://render.com)
2. Sign up with GitHub (recommended) or email
3. Verify your email if required

### Step 2: Create New Web Service
1. Click **"New +"** button in dashboard
2. Select **"Web Service"**
3. Connect your GitHub repository
4. Select the **StockView** repository

### Step 3: Configure Backend Service

**Basic Settings:**
- **Name**: `stockview-backend` (or your preferred name)
- **Region**: Choose closest to your users
- **Branch**: `main` (or your default branch)
- **Root Directory**: `backend`
- **Runtime**: `Python 3`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`

**Environment Variables:**
- No environment variables needed for basic setup
- Render automatically sets `$PORT` variable

**Plan:**
- Select **Free** plan (or paid if you prefer)

### Step 4: Deploy
1. Click **"Create Web Service"**
2. Wait for build to complete (5-10 minutes first time)
3. Note your service URL (e.g., `https://stockview-5ppc.onrender.com`)

### Step 5: Verify Backend
1. Visit your backend URL: `https://your-backend.onrender.com/`
2. Should see: `{"status":"healthy","message":"StockView API is running"}`
3. Test an endpoint: `https://your-backend.onrender.com/price?symbol=AAPL`
4. Should return stock price data (or 404 if market is closed)

**✅ Backend is ready when:**
- Health check returns 200 OK
- `/price` endpoint works (may return 404 if market closed, but no 500 errors)

---

## 🎨 Part 2: Frontend Deployment on Vercel

### Step 1: Create Vercel Account
1. Go to [vercel.com](https://vercel.com)
2. Sign up with GitHub (recommended)
3. Import your GitHub repository

### Step 2: Import Project
1. Click **"Add New..."** → **"Project"**
2. Select your **StockView** repository
3. Click **"Import"**

### Step 3: Configure Frontend

**Framework Preset:**
- Vercel should auto-detect **Vite**
- If not, select **"Other"**

**Root Directory:**
- Click **"Edit"** next to Root Directory
- Set to: `frontend`
- Click **"Continue"**

**Build Settings:**
- **Build Command**: `npm run build` (auto-detected)
- **Output Directory**: `dist` (auto-detected)
- **Install Command**: `npm install` (auto-detected)

**Environment Variables:**
- Click **"Environment Variables"**
- Add new variable:
  - **Key**: `VITE_API_URL`
  - **Value**: `https://your-backend.onrender.com` (use your actual Render backend URL)
  - **Environment**: Select all (Production, Preview, Development)
- Click **"Save"**

### Step 4: Deploy
1. Click **"Deploy"**
2. Wait for build to complete (2-5 minutes)
3. Vercel will provide your frontend URL

### Step 5: Verify Frontend
1. Visit your Vercel URL
2. Open browser console (F12)
3. Check for:
   - `API URL: https://your-backend.onrender.com` (should show your backend URL)
   - No CORS errors
   - No 500 errors from backend

**✅ Frontend is ready when:**
- Page loads without errors
- Can search for stocks (e.g., "AAPL")
- Data displays correctly

---

## 🔥 Part 3: Firebase Firestore Setup

### Step 1: Access Firebase Console
1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Select your project: `stockview-dashboard`
3. If project doesn't exist, create it first

### Step 2: Enable Firestore
1. Click **"Firestore Database"** in left sidebar
2. If not enabled, click **"Create database"**
3. Select **"Start in test mode"** (we'll add rules next)
4. Choose a location (closest to your users)
5. Click **"Enable"**

### Step 3: Deploy Security Rules
1. In Firestore Database, click **"Rules"** tab
2. Copy the contents of `firestore.rules` from your repository
3. Paste into the Rules editor
4. Click **"Publish"**

**Rules should look like:**
```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    match /watchlist/{document} {
      allow read, write: if true;
    }
    match /{document=**} {
      allow read, write: if false;
    }
  }
}
```

### Step 4: Verify Firestore
1. Go back to your frontend
2. Try adding a stock to watchlist
3. Should work without permission errors

**✅ Firestore is ready when:**
- Can add stocks to watchlist
- No "Missing or insufficient permissions" errors

---

## ✅ Final Verification Checklist

### Backend (Render)
- [ ] Health endpoint works: `https://your-backend.onrender.com/`
- [ ] Price endpoint works: `https://your-backend.onrender.com/price?symbol=AAPL`
- [ ] History endpoint works: `https://your-backend.onrender.com/history?symbol=AAPL&range=1d&interval=5m`
- [ ] No 500 errors in Render logs

### Frontend (Vercel)
- [ ] Frontend loads at Vercel URL
- [ ] Can search for stocks
- [ ] Charts display correctly
- [ ] No console errors
- [ ] API calls go to correct backend URL

### Firestore
- [ ] Can add stocks to watchlist
- [ ] Can remove stocks from watchlist
- [ ] Watchlist persists across page refreshes
- [ ] No permission errors

---

## 🐛 Troubleshooting

### Backend Issues

**Problem: Backend returns 500 errors**
- Check Render logs for specific error messages
- Verify all dependencies in `requirements.txt`
- Ensure Python version matches `runtime.txt` (3.11.10)

**Problem: "Impersonating chrome136 is not supported"**
- This should be fixed with the curl_cffi mock
- If still occurring, check that `backend/main.py` has the mock code

**Problem: Backend sleeps (free tier)**
- Free tier services sleep after 15 minutes of inactivity
- First request after sleep takes 30-50 seconds
- Consider upgrading to paid plan for always-on service

### Frontend Issues

**Problem: API calls go to localhost**
- Verify `VITE_API_URL` is set in Vercel environment variables
- Redeploy frontend after adding environment variable
- Check browser console for `API URL: ...` log

**Problem: CORS errors**
- Backend CORS is configured to allow all origins
- If issues persist, check Render logs for CORS errors

**Problem: 404 errors on routes**
- Verify `vercel.json` is in `frontend/` directory
- Check that build output is `dist/`

### Firestore Issues

**Problem: "Missing or insufficient permissions"**
- Verify Firestore rules are deployed
- Check that rules allow read/write to `watchlist` collection
- Ensure you're using the correct Firebase project

---

## 📝 Important Notes

### Free Tier Limitations

**Render (Backend):**
- Services sleep after 15 minutes of inactivity
- 750 build minutes/month
- 30-50 second spin-up time when sleeping

**Vercel (Frontend):**
- 100GB bandwidth/month
- 6000 build minutes/month
- Unlimited deployments

### Security Considerations

**Current Setup:**
- Firestore rules allow public read/write (for development)
- Backend CORS allows all origins
- API keys are in code (consider moving to environment variables)

**For Production:**
- Add authentication to Firestore rules
- Restrict CORS to your Vercel domain
- Move API keys to environment variables

---

## 🎉 Success!

If all checkboxes are complete, your StockView application is fully deployed and ready to use!

**Your URLs:**
- Frontend: `https://your-app.vercel.app`
- Backend: `https://your-backend.onrender.com`
- Firebase: `https://console.firebase.google.com/project/stockview-dashboard`

---

## 📞 Need Help?

1. Check Render logs: Dashboard → Your Service → Logs
2. Check Vercel logs: Dashboard → Your Project → Deployments → View Function Logs
3. Check browser console: F12 → Console tab
4. Review error messages carefully - they usually point to the issue

