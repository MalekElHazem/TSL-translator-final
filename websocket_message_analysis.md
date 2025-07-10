# WebSocket Message Size Analysis

## Current Image Processing

Based on your ExpoApp.js configuration:
- Camera resolution: Default (likely 1920x1080 or similar)
- Frame rate: 5 FPS (200ms intervals)
- Format: Base64 encoded JPEG

## Estimated Message Sizes

### Typical Base64 Image Sizes:
- **Low quality JPEG**: 20-50 KB
- **Medium quality JPEG**: 50-150 KB  
- **High quality JPEG**: 150-500 KB
- **Very high quality**: 500KB - 1MB+

### Your JSON Message Structure:
```json
{
  "type": "frame",
  "data": "base64_image_data_here...",
  "timestamp": 1234567890
}
```

**Additional overhead**: ~100 bytes for JSON structure

## Potential Issues

### 1. Default Limit Risk
- Default websockets limit: **1 MB**
- High-quality images could approach this limit
- Risk of message rejection

### 2. Network Performance
- Larger messages = higher latency
- Mobile data usage concerns
- Potential connection drops

## Recommendations

### Option 1: Set Explicit Limits (Conservative)
```python
async with websockets.serve(
    handle_client, 
    "0.0.0.0", 
    8765,
    max_size=2 * 1024 * 1024,  # 2MB limit
    ping_interval=30,
    ping_timeout=30
):
```

### Option 2: Optimize Image Quality
```javascript
// In your mobile app
const imageOptions = {
  quality: 0.6,        // Reduce quality to 60%
  base64: true,
  skipProcessing: true
};
```

### Option 3: Image Compression
```python
# Server-side compression check
def compress_if_needed(base64_data):
    if len(base64_data) > 500000:  # 500KB
        # Decode, compress, re-encode
        pass
```

## Performance Monitoring

Add message size logging:
```python
message_size = len(message_data)
if message_size > 500000:  # 500KB
    logger.warning(f"Large message: {message_size} bytes")
```
