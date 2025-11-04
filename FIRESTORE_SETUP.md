# Firestore Security Rules Setup

## Issue
The application was getting "FirebaseError: Missing or insufficient permissions" when trying to access the Firestore `watchlist` collection.

## Solution

You need to deploy security rules to your Firestore database that allow read/write access to the `watchlist` collection.

### Step 1: Deploy Firestore Security Rules

1. **Go to Firebase Console**:
   - Visit [Firebase Console](https://console.firebase.google.com/)
   - Select your project: `stockview-dashboard`

2. **Navigate to Firestore Database**:
   - Click on "Firestore Database" in the left sidebar
   - Click on the "Rules" tab

3. **Update the Rules**:
   - Copy the contents of `firestore.rules` file in this repository
   - Paste it into the Firebase Console Rules editor
   - Click "Publish" to deploy the rules

### Step 2: Verify Rules

The rules should look like this:

```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    // Allow read/write access to watchlist collection for all users
    // Note: For production, consider adding authentication checks
    match /watchlist/{document} {
      allow read, write: if true;
    }
    
    // Default deny rule for other collections
    match /{document=**} {
      allow read, write: if false;
    }
  }
}
```

### Step 3: Test the Application

After deploying the rules:
1. Refresh your frontend application
2. Try adding a stock to the watchlist
3. The permission error should be resolved

## Security Note

**Current Rules**: The rules allow public read/write access to the `watchlist` collection. This is fine for development, but for production, consider:

1. **Adding Authentication**: Require users to sign in before accessing the watchlist
2. **User-specific Watchlists**: Store watchlists per user ID
3. **Rate Limiting**: Implement rate limiting on writes

Example of user-specific watchlist rules:

```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    match /watchlist/{userId}/{document} {
      allow read, write: if request.auth != null && request.auth.uid == userId;
    }
  }
}
```

## Alternative: Using Firebase CLI

If you have Firebase CLI installed, you can deploy rules directly:

```bash
firebase deploy --only firestore:rules
```

Make sure you have `firestore.rules` in your project root and it's configured in `firebase.json`.
