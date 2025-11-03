# 🚀 StockView Deployment Guide

This guide will help you deploy your StockView application with advanced ML features to free hosting platforms.

## 📋 Prerequisites

- GitHub account
- Render account (free tier)
- Vercel account (free tier)
- Git installed on your machine

## 🔧 Backend Deployment (Render)

### Step 1: Prepare Your Repository

1. **Initialize Git** (if not already done):
   ```bash
   git init
   git add .
   git commit -m "Initial commit with ML features"
   ```

2. **Create GitHub Repository**:
   - Go to [GitHub](https://github.com)
   - Create a new repository named `stockview-ml`
   - Push your code:
     ```bash
     git remote add origin https://github.com/YOUR_USERNAME/stockview-ml.git
     git branch -M main
     git push -u origin main
     ```

### Step 2: Deploy to Render

1. **Sign up for Render**:
   - Go to [Render.com](https://render.com)
   - Sign up with your GitHub account

2. **Create New Web Service**:
   - Click "New +" in the dashboard
   - Select "Web Service"
   - Choose "Build and deploy from a Git repository"
   - Connect your GitHub account and select your `stockview-ml` repository

3. **Configure Service Settings**:
   - **Name**: `stockview-backend` (or any name you prefer)
   - **Region**: Choose closest to your users (e.g., `Oregon (US West)`)
   - **Branch**: `main`
   - **Root Directory**: Leave blank (or set to `backend` if using manual deployment)
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r backend/requirements.txt`
   - **Start Command**: `cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Select `Free` plan

4. **Alternative: Use render.yaml**:
   - Render can auto-detect `render.yaml` in your repository
   - The provided `render.yaml` will automatically configure your service
   - Just select your repository and Render will use the configuration

5. **Deploy**:
   - Click "Create Web Service"
   - Render will automatically build and deploy your backend
   - Wait for deployment to complete (5-10 minutes for first build)
   - Note down your backend URL (e.g., `https://stockview-backend.onrender.com`)

## 🎨 Frontend Deployment (Vercel)

### Step 1: Prepare Frontend

1. **Update API Endpoints**:
   - Replace `http://localhost:8000` with your Render backend URL
   - Update all API calls in `frontend/src/App.jsx`

2. **Create Vercel Configuration**:
   - Create `vercel.json` in the frontend folder

### Step 2: Deploy to Vercel

1. **Sign up for Vercel**:
   - Go to [Vercel.com](https://vercel.com)
   - Sign up with your GitHub account

2. **Import Project**:
   - Click "New Project"
   - Import your GitHub repository
   - Set root directory to `frontend`
   - Vercel will auto-detect it's a React app

3. **Deploy**:
   - Click "Deploy"
   - Wait for deployment (2-3 minutes)
   - Get your frontend URL (e.g., `https://your-app.vercel.app`)

## 🔗 Connect Frontend to Backend

### Update API Endpoints

Replace all instances of `http://localhost:8000` in your frontend with your Render backend URL:

```javascript
// In frontend/src/App.jsx, replace:
const response = await fetch(`http://localhost:8000/predict?...`);

// With:
const response = await fetch(`https://your-backend.onrender.com/predict?...`);
```

### Environment Variables (Optional)

For better security, you can use environment variables:

1. **In Vercel**:
   - Go to your project settings
   - Add environment variable: `REACT_APP_API_URL=https://your-backend.onrender.com`

2. **In Frontend Code**:
   ```javascript
   const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
   ```

## 🧪 Testing Your Deployment

### Backend Testing

Test your Render backend:
```bash
curl https://your-backend.onrender.com/
curl https://your-backend.onrender.com/predict?symbol=AAPL&algorithm=random_forest
```

### Frontend Testing

1. Visit your Vercel URL
2. Search for a stock (e.g., AAPL)
3. Test predictions with different algorithms
4. Verify algorithm comparison works

## 📊 Free Tier Limits

### Render (Backend)
- **Monthly Usage**: Free tier available
- **Build Time**: 750 minutes/month
- **Deployments**: Unlimited
- **Sleep**: Apps sleep after 15 minutes of inactivity (free tier)
- **Spin-up Time**: Free tier apps may take 30-50 seconds to spin up when sleeping

### Vercel (Frontend)
- **Bandwidth**: 100GB/month
- **Build Time**: 6000 minutes/month
- **Deployments**: Unlimited
- **Custom Domains**: 1 free

## 🔧 Troubleshooting

### Common Issues

1. **Backend Not Starting**:
   - Check Render logs for errors (available in the Render dashboard)
   - Ensure all dependencies are in `requirements.txt`
   - Verify `Procfile` or `render.yaml` configuration is correct
   - Check that `$PORT` environment variable is used (Render sets this automatically)

2. **CORS Errors**:
   - Backend CORS is set to allow all origins
   - If issues persist, update CORS settings in `main.py`

3. **API Calls Failing**:
   - Verify backend URL is correct
   - Check if backend is running (visit health check endpoint)
   - Ensure HTTPS is used in production

4. **Build Failures**:
   - Check dependency versions in `requirements.txt`
   - Ensure Python version compatibility
   - Review build logs for specific errors

### Performance Optimization

1. **Backend**:
   - Models are trained on each request (consider caching)
   - Use smaller datasets for faster training
   - Consider model persistence for production

2. **Frontend**:
   - Enable Vercel's automatic optimizations
   - Use React.memo for expensive components
   - Implement loading states for better UX

## 🚀 Production Considerations

### Security
- Replace API keys with environment variables
- Implement rate limiting
- Add input validation
- Use HTTPS everywhere

### Performance
- Implement caching for predictions
- Use CDN for static assets
- Optimize ML model training
- Add database for persistent storage

### Monitoring
- Set up error tracking (Sentry)
- Monitor API usage
- Track performance metrics
- Set up alerts for downtime

## 📞 Support

If you encounter issues:
1. Check Render/Vercel logs
2. Review this deployment guide
3. Test locally first
4. Check GitHub issues for similar problems
5. Verify render.yaml configuration is correct

## 🎉 Success!

Once deployed, you'll have:
- ✅ Backend API running on Render
- ✅ Frontend app running on Vercel
- ✅ Advanced ML predictions working
- ✅ Algorithm comparison features
- ✅ Free hosting with good performance

**Note**: Render's free tier services spin down after 15 minutes of inactivity. The first request after spin-down may take 30-50 seconds to respond while the service starts up. This is normal for free tier hosting.

Your StockView application is now live and accessible worldwide! 🌍
