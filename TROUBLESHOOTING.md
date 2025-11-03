# Troubleshooting: "Stock not found or API error NetworkError"

## Quick Diagnosis Steps

### 1. Check Browser Console
1. Open your deployed Vercel site
2. Press `F12` to open Developer Tools
3. Go to the **Console** tab
4. Look for:
   - `API URL: ...` - This shows what URL the frontend is using
   - `Environment variable: ...` - This shows if VITE_API_URL is set
   - Any red error messages

### 2. Check Network Tab
1. In Developer Tools, go to the **Network** tab
2. Try searching for a stock (e.g., "AAPL")
3. Look for failed requests:
   - **Red requests** = Failed
   - Check the **URL** - is it pointing to `localhost:8000` or your Render backend?
   - Check the **Status** - 404, 500, CORS error, etc.

### 3. Verify Vercel Environment Variable

**In Vercel Dashboard:**
1. Go to your project → **Settings** → **Environment Variables**
2. Check if `VITE_API_URL` exists:
   - **Key**: `VITE_API_URL`
   - **Value**: Should be your Render backend URL (e.g., `https://stockview-backend.onrender.com`)
   - **Environment**: Should be checked for Production, Preview, and Development

3. **If missing or incorrect:**
   - Add/Update the variable
   - **Redeploy** your site (Settings → Redeploy)

### 4. Verify Render Backend is Running

**In Render Dashboard:**
1. Check your backend service status
2. Open the **Logs** tab
3. Verify the service is running (not sleeping)
4. Test the health endpoint:
   - Visit: `https://your-backend.onrender.com/`
   - Should return: `{"status":"healthy","message":"StockView API is running"}`

### 5. Test Backend API Directly

Test your backend endpoints directly in a browser or using curl:

```bash
# Test health endpoint
curl https://your-backend.onrender.com/

# Test price endpoint
curl https://your-backend.onrender.com/price?symbol=AAPL

# Test history endpoint
curl https://your-backend.onrender.com/history?symbol=AAPL&range=1d&interval=5m
```

If these fail, the issue is with the backend, not the frontend.

## Common Issues and Solutions

### Issue 1: API URL shows `localhost:8000`
**Problem**: `VITE_API_URL` is not set in Vercel  
**Solution**: 
1. Add `VITE_API_URL` = `https://your-backend.onrender.com` in Vercel
2. Redeploy the site

### Issue 2: CORS Error
**Problem**: Backend CORS settings not allowing Vercel domain  
**Solution**: 
- Your backend already has `allow_origins=["*"]` which should work
- If still having issues, check Render logs for CORS errors

### Issue 3: Backend is Sleeping (Free Tier)
**Problem**: Render free tier services sleep after 15 minutes  
**Solution**: 
- First request after sleep takes 30-50 seconds
- Wait for the backend to wake up
- Or upgrade to a paid plan to keep it always running

### Issue 4: Network Error / Failed to Fetch
**Problem**: Cannot connect to backend  
**Solution**:
1. Verify backend URL is correct (check Render dashboard)
2. Verify backend is running (check Render logs)
3. Check if backend URL uses `https://` (not `http://`)
4. Make sure there's no trailing slash in the URL

### Issue 5: 404 or 500 Errors
**Problem**: Backend endpoint not found or server error  
**Solution**:
1. Check Render logs for specific error messages
2. Verify the backend routes match what the frontend expects
3. Test the backend directly using curl

## Testing Checklist

- [ ] Backend is running on Render (check dashboard)
- [ ] Backend health endpoint works: `https://your-backend.onrender.com/`
- [ ] `VITE_API_URL` is set in Vercel environment variables
- [ ] `VITE_API_URL` value matches your Render backend URL
- [ ] Vercel site has been redeployed after setting environment variable
- [ ] Browser console shows the correct API URL (not localhost)
- [ ] Network tab shows requests going to Render backend (not localhost)

## Still Not Working?

1. **Check the browser console** - The updated code now shows detailed error messages
2. **Check Render logs** - Look for errors in the backend
3. **Verify URLs match** - Make sure Vercel `VITE_API_URL` matches your Render backend URL exactly
4. **Test backend directly** - Use curl to verify backend works independently

