# TSL Expo Mobile App Setup Guide

## Step 1: Create New Expo Project
```bash
npx create-expo-app TSLTranslator
cd TSLTranslator
```

## Step 2: Install Required Dependencies
```bash
# Core dependencies for camera and WebSocket
npx expo install expo-camera expo-media-library expo-av

# For file system operations
npx expo install expo-file-system

# For permissions
npx expo install expo-permissions

# WebSocket is built into React Native - no install needed!
```

## Step 3: Update app.json for permissions
```json
{
  "expo": {
    "name": "TSL Translator",
    "slug": "tsl-translator",
    "version": "1.0.0",
    "platforms": ["ios", "android"],
    "permissions": [
      "CAMERA",
      "RECORD_AUDIO"
    ],
    "ios": {
      "infoPlist": {
        "NSCameraUsageDescription": "This app needs camera access for sign language detection"
      }
    },
    "android": {
      "permissions": [
        "android.permission.CAMERA",
        "android.permission.RECORD_AUDIO"
      ]
    }
  }
}
```
