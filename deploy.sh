#!/bin/bash

# StockView Deployment Script
echo "🚀 Starting StockView Deployment Process..."

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "📦 Initializing Git repository..."
    git init
    git add .
    git commit -m "Initial commit with ML features"
fi

# Check if remote origin exists
if ! git remote get-url origin > /dev/null 2>&1; then
    echo "⚠️  No remote origin found. Please add your GitHub repository:"
    echo "   git remote add origin https://github.com/YOUR_USERNAME/stockview-ml.git"
    echo "   git branch -M main"
    echo "   git push -u origin main"
    exit 1
fi

# Push to GitHub
echo "📤 Pushing to GitHub..."
git add .
git commit -m "Deploy StockView with ML features" || echo "No changes to commit"
git push origin main

echo "✅ Code pushed to GitHub successfully!"
echo ""
echo "🔧 Next Steps:"
echo "1. Go to https://render.com and deploy your backend"
echo "2. Go to https://vercel.com and deploy your frontend"
echo "3. Update frontend API endpoints with your Render backend URL"
echo ""
echo "📖 See DEPLOYMENT_GUIDE.md for detailed instructions"
