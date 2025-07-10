# Quick WebSocket Setup for Mobile App

## What You Have Running:
✅ WebSocket Server on: ws://192.168.1.11:8765
✅ FastAPI Server on: http://192.168.1.11:8000

## Test Steps:

### Step 1: Test WebSocket Connection (Desktop)
1. Open `simple_websocket_test.html` in your browser
2. Click "Connect" - you should see "Connected ✅"
3. Click "Ping Server" - you should get a response
4. Check your server terminal for connection logs

### Step 2: Add to React Native App
1. Copy `mobile_websocket_client.js` to your React Native project
2. In your App.js:
```javascript
import TSLWebSocketClient from './mobile_websocket_client.js';

export default function App() {
  return <TSLWebSocketClient />;
}
```

### Step 3: Update IP Address
In `mobile_websocket_client.js`, line 11:
```javascript
const SERVER_URL = 'ws://192.168.1.11:8765'; // Your computer's IP
```

### Step 4: Test Mobile Connection
1. Make sure phone and computer are on same Wi-Fi
2. Start React Native app
3. Tap "Connect" - should show "Connected ✅"
4. Tap "Test Ping" - should get server response
5. Tap "Send Test Frame" - should trigger server processing

## Troubleshooting:

### If Connection Fails:
1. Check Windows Firewall (allow Python/port 8765)
2. Verify IP address with `ipconfig`
3. Ensure both devices on same Wi-Fi network
4. Check server terminal for error messages

### Server Logs to Watch For:
- "Client connected: [IP]:[PORT]"
- "Client disconnected: [IP]:[PORT]"
- WebSocket message processing logs

### Mobile App Expected Behavior:
- Connect: Shows "Connected ✅"
- Ping: Shows "Server responded to ping ✅"  
- Test Frame: Shows "Collecting frames..." messages
- Sensitivity: Adjusts motion detection threshold

## Next Steps:
Once basic connection works, you can:
1. Add camera functionality 
2. Capture and send real video frames
3. Display sign language predictions
4. Handle reconnection and errors

## Current Limitations:
- Test frame is dummy data
- No camera integration yet
- Basic error handling
- No offline support

The WebSocket server is designed for real-time streaming, but you can also use the REST API at http://192.168.1.11:8000/predict_video for recorded videos.
