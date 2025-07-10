# Complete Expo Go Setup Instructions

## 🚀 Quick Start Guide

### 1. Create New Expo Project
```bash
npx create-expo-app TSLTranslator
cd TSLTranslator
```

### 2. Install Dependencies
```bash
# Camera and media
npx expo install expo-camera expo-file-system

# No need to install WebSocket - it's built into React Native!
```

### 3. Project Structure
```
TSLTranslator/
├── App.js                    # Main app component
├── services/
│   └── TSLWebSocketService.js # WebSocket service
├── app.json                  # Expo configuration
└── package.json
```

### 4. Copy Files
1. Copy `TSLWebSocketService.js` to `services/TSLWebSocketService.js`
2. Replace `App.js` with the provided `ExpoApp.js` content
3. Update `app.json` with camera permissions

### 5. Update app.json
```json
{
  "expo": {
    "name": "TSL Translator",
    "slug": "tsl-translator",
    "version": "1.0.0",
    "orientation": "portrait",
    "platforms": ["ios", "android"],
    "permissions": ["CAMERA"],
    "ios": {
      "infoPlist": {
        "NSCameraUsageDescription": "Camera access is required for sign language detection"
      }
    },
    "android": {
      "permissions": ["android.permission.CAMERA"]
    }
  }
}
```

### 6. Update Server IP
In `App.js`, change line:
```javascript
const SERVER_URL = 'ws://192.168.1.11:8765'; // Your computer's IP
```

### 7. Start Development
```bash
npx expo start
```

### 8. Run on Device
- Install **Expo Go** app on your phone
- Scan QR code from terminal
- Grant camera permissions when prompted

## 📱 App Features

### ✅ What the App Includes:
- **WebSocket Connection** to your server
- **Real-time Camera** feed with front camera
- **Frame Capture** and transmission (5 FPS)
- **Live Predictions** display with confidence scores
- **Connection Status** indicator
- **Manual Controls** for ping, sensitivity adjustment
- **Auto-reconnection** when connection drops
- **Error Handling** with user-friendly messages

### 🎯 How to Use:
1. **Connect** - Tap "Connect" button
2. **Grant Permissions** - Allow camera access
3. **Start Detection** - Tap "Start Detection"
4. **Perform Signs** - Make sign language gestures
5. **View Results** - See predictions in real-time

## 🔧 Troubleshooting

### Common Issues:
1. **"Connection failed"** 
   - Ensure both devices on same WiFi
   - Check server IP address
   - Verify WebSocket server is running

2. **"Camera permission denied"**
   - Go to device Settings > Apps > Expo Go > Permissions
   - Enable Camera permission

3. **"No predictions"**
   - Check server logs for frame processing
   - Adjust motion sensitivity
   - Ensure good lighting conditions

4. **App crashes**
   - Check Expo Go app is updated
   - Restart Expo development server
   - Clear Expo Go cache

### Performance Tips:
- **Close other apps** for better performance
- **Use good lighting** for better hand detection
- **Hold phone steady** for consistent frames
- **Adjust sensitivity** if too sensitive/not sensitive enough

## 📊 Expected Behavior:
- **Connection**: Shows green status when connected
- **Frame capture**: Overlay shows frame count increasing
- **Server communication**: Status updates show frame collection progress
- **Predictions**: Results appear with confidence percentages
- **Reconnection**: Automatically reconnects if connection drops

## 🎉 You're Ready!
Your Expo Go app is now configured for real-time sign language detection via WebSocket!
