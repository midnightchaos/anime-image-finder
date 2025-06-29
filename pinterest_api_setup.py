#!/usr/bin/env python3
"""
Pinterest API Setup Helper
Helps you get set up with Pinterest API access
"""

import os
import sys
import requests
import json
from pathlib import Path
import webbrowser
from urllib.parse import urlencode

def create_privacy_policy():
    """Create a simple privacy policy file"""
    privacy_policy = """
# Privacy Policy for Anime Image Finder

## What we collect
- We do not collect any personal information
- We only process images you provide locally
- We do not store or transmit your images

## How we use data
- Images are processed locally on your computer
- We only send search queries to Pinterest
- No image data is stored on our servers

## Data sharing
- We do not share any data with third parties
- We only interact with Pinterest's public API
- All processing happens on your local machine

## Contact
This is a local application with no data collection.
"""
    
    with open("privacy_policy.md", "w") as f:
        f.write(privacy_policy)
    
    print("✅ Created privacy_policy.md")
    return "privacy_policy.md"

def create_app_description():
    """Create app description for Pinterest"""
    app_description = """
Anime Image Finder - A local application that helps find similar anime images on Pinterest.

Features:
- Processes local anime images
- Finds visually similar images on Pinterest
- Pins similar images to your Pinterest boards
- All processing done locally on your computer

This app does not collect, store, or transmit any personal data.
"""
    
    with open("app_description.txt", "w") as f:
        f.write(app_description)
    
    print("✅ Created app_description.txt")
    return "app_description.txt"

def get_oauth_url(client_id, redirect_uri="http://localhost:8080/callback"):
    """Generate OAuth URL for Pinterest"""
    scopes = [
        "boards:read",
        "boards:write", 
        "pins:read",
        "pins:write"
    ]
    
    params = {
        'client_id': client_id,
        'redirect_uri': redirect_uri,
        'scope': ','.join(scopes),
        'response_type': 'code'
    }
    
    oauth_url = f"https://www.pinterest.com/oauth/?{urlencode(params)}"
    return oauth_url

def exchange_code_for_token(client_id, client_secret, auth_code, redirect_uri="http://localhost:8080/callback"):
    """Exchange authorization code for access token"""
    url = "https://api.pinterest.com/v5/oauth/token"
    
    data = {
        'grant_type': 'authorization_code',
        'code': auth_code,
        'redirect_uri': redirect_uri,
        'client_id': client_id,
        'client_secret': client_secret
    }
    
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    
    response = requests.post(url, data=data, headers=headers)
    
    if response.status_code == 200:
        token_data = response.json()
        return token_data.get('access_token')
    else:
        print(f"❌ Error exchanging code for token: {response.status_code}")
        print(f"Response: {response.text}")
        return None

def test_api_connection(access_token):
    """Test Pinterest API connection"""
    url = "https://api.pinterest.com/v5/user_account/boards"
    headers = {
        'Authorization': f'Bearer {access_token}'
    }
    
    try:
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            boards = data.get('items', [])
            print(f"✅ API connection successful!")
            print(f"📋 Found {len(boards)} boards:")
            for board in boards[:5]:  # Show first 5 boards
                print(f"  - {board.get('name', 'Unknown')}")
            return True
        else:
            print(f"❌ API connection failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing API: {e}")
        return False

def main():
    """Main setup function"""
    print("🎯 Pinterest API Setup Helper")
    print("=" * 50)
    
    # Step 1: Create required files
    print("\n📝 Step 1: Creating required files...")
    privacy_file = create_privacy_policy()
    description_file = create_app_description()
    
    # Step 2: Instructions
    print("\n📋 Step 2: Pinterest Developer Setup")
    print("1. Go to: https://developers.pinterest.com/")
    print("2. Sign in with your Pinterest account")
    print("3. Click 'Create App'")
    print("4. Fill in the details:")
    print(f"   - App Name: Anime Image Finder")
    print(f"   - App Description: (use content from {description_file})")
    print(f"   - App Type: Web App")
    print(f"   - Redirect URIs: http://localhost:8080/callback")
    print(f"   - Privacy Policy URL: (upload {privacy_file})")
    print("5. Submit the app")
    
    # Step 3: Get credentials
    print("\n🔑 Step 3: Get Your Credentials")
    client_id = input("Enter your App ID: ").strip()
    client_secret = input("Enter your App Secret: ").strip()
    
    if not client_id or not client_secret:
        print("❌ App ID and App Secret are required!")
        return
    
    # Step 4: Generate OAuth URL
    print("\n🔗 Step 4: Get Authorization Code")
    oauth_url = get_oauth_url(client_id)
    print(f"OAuth URL: {oauth_url}")
    
    # Open browser
    try:
        webbrowser.open(oauth_url)
        print("🌐 Opened browser for authorization...")
    except:
        print("Please manually open the OAuth URL in your browser")
    
    print("\n📋 Instructions:")
    print("1. Authorize the app in your browser")
    print("2. Copy the authorization code from the redirect URL")
    print("3. Paste it below")
    
    # Step 5: Exchange code for token
    auth_code = input("\nEnter the authorization code: ").strip()
    
    if not auth_code:
        print("❌ Authorization code is required!")
        return
    
    print("\n🔄 Exchanging code for access token...")
    access_token = exchange_code_for_token(client_id, client_secret, auth_code)
    
    if access_token:
        print(f"✅ Access token obtained!")
        print(f"Token: {access_token[:20]}...")
        
        # Step 6: Test API
        print("\n🧪 Step 5: Testing API Connection...")
        if test_api_connection(access_token):
            # Save token
            with open(".env", "w") as f:
                f.write(f"PINTEREST_ACCESS_TOKEN={access_token}\n")
            
            print(f"\n✅ Setup complete!")
            print(f"📁 Access token saved to .env file")
            print(f"🚀 You can now run: python pinterest_pinner.py")
        else:
            print("❌ API test failed. Please check your credentials.")
    else:
        print("❌ Failed to get access token. Please try again.")

if __name__ == "__main__":
    main() 