# Fixes Applied for Errors

## Issues Fixed

### 1. "Impersonating chrome136 is not supported" Error

**Problem**: yfinance was trying to use `curl_cffi` to impersonate Chrome 136, which isn't supported, causing 500 errors on all API endpoints.

**Solution Applied**:
- Added environment variable `YFINANCE_DISABLE_CURL_CFFI=1` before importing yfinance
- Mocked `curl_cffi` module in `sys.modules` to prevent yfinance from detecting/using it
- Explicitly passing `session=requests.Session()` to all `yf.Ticker()` instances

**Files Modified**:
- `backend/main.py`: Added curl_cffi prevention logic and ensured all Ticker instances use requests.Session()

**Next Steps**:
1. Redeploy your backend on Render
2. The changes should prevent curl_cffi from being used

### 2. Firestore "Missing or insufficient permissions" Error

**Problem**: Firestore security rules were blocking read/write access to the `watchlist` collection.

**Solution Applied**:
- Created `firestore.rules` file with security rules allowing public read/write to the `watchlist` collection

**Files Created**:
- `firestore.rules`: Security rules for Firestore
- `FIRESTORE_SETUP.md`: Instructions for deploying Firestore rules

**Next Steps**:
1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Select your project: `stockview-dashboard`
3. Navigate to Firestore Database → Rules tab
4. Copy the contents of `firestore.rules` and paste into the Rules editor
5. Click "Publish" to deploy the rules

## Testing After Deployment

1. **Backend API**:
   - Test `/price?symbol=AAPL` - should return 200 OK
   - Test `/history?symbol=AAPL&range=1d&interval=5m` - should return 200 OK
   - Should no longer see "Impersonating chrome136 is not supported" errors

2. **Firestore**:
   - Try adding a stock to the watchlist
   - Should no longer see "Missing or insufficient permissions" errors

## Notes

- The curl_cffi mocking approach prevents yfinance from detecting curl_cffi even if it's installed
- All yfinance Ticker instances explicitly use `requests.Session()` to ensure compatibility
- Firestore rules are set to allow public access - for production, consider adding authentication

