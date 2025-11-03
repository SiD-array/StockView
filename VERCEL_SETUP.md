# Vercel Configuration for StockView Frontend

## Required Changes in Vercel Dashboard

### Step 1: Add Environment Variable

1. **Go to your Vercel project dashboard**
   - Navigate to your project on [vercel.com](https://vercel.com)
   - Click on your project

2. **Open Settings → Environment Variables**
   - Click on "Settings" tab
   - Click on "Environment Variables" in the left sidebar

3. **Add the Backend API URL**
   - **Key**: `VITE_API_URL`
   - **Value**: `https://your-backend.onrender.com` (replace with your actual Render backend URL)
   - **Environment**: Select all environments (Production, Preview, Development)
   - Click "Save"

### Step 2: Verify Build Settings

1. **Go to Settings → General**
   - **Root Directory**: Should be set to `frontend` (if your project structure requires it)
   - **Build Command**: Should be `npm run build` (or `npm run build` if using npm)
   - **Output Directory**: Should be `dist` (for Vite projects)

2. **Framework Preset**: Should be set to "Vite" or "Other" (Vercel should auto-detect Vite)

### Step 3: Redeploy

After adding the environment variable:
1. **Trigger a new deployment**:
   - Go to the "Deployments" tab
   - Click the three dots (⋯) on the latest deployment
   - Click "Redeploy"
   - Or push a new commit to trigger auto-deploy

## Important Notes

- **Environment Variable Name**: Must be `VITE_API_URL` (not `REACT_APP_API_URL`)
  - Vite uses the `VITE_` prefix for environment variables
  - This is already configured in your `App.jsx` file

- **Backend URL Format**: 
  - Your Render backend URL will look like: `https://stockview-backend.onrender.com`
  - Make sure to use `https://` (not `http://`)
  - Don't include a trailing slash

- **CORS**: Your backend already has CORS configured to allow all origins, so the frontend should work without additional CORS configuration

## Testing

After deployment, test your frontend:
1. Visit your Vercel URL
2. Try searching for a stock (e.g., "AAPL")
3. Verify the API calls are working by checking the browser console (F12 → Network tab)
4. Make sure requests are going to your Render backend URL

## Troubleshooting

- **API calls failing**: Check that `VITE_API_URL` is set correctly in Vercel
- **CORS errors**: Verify your Render backend is running and CORS is enabled
- **Build errors**: Check that the root directory is set correctly (should be `frontend` if your repo root is above frontend)

