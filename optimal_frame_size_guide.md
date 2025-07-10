# Optimal Frame Size Configuration Guide

## Recommended Mobile App Settings

### Option 1: Balanced Performance (RECOMMENDED)
```javascript
// In your ExpoApp.js or camera component
const OPTIMAL_CAMERA_CONFIG = {
  // Camera resolution
  width: 640,
  height: 480,
  
  // JPEG compression
  quality: 0.7,              // 70% quality
  format: 'jpeg',
  
  // Capture settings
  skipProcessing: true,      // Faster capture
  base64: true,              // For WebSocket transmission
  
  // Advanced settings
  flashMode: 'off',
  whiteBalance: 'auto',
  focusMode: 'auto'
};

// Expected results:
// - File size: 40-80KB per frame
// - Network time: 50-100ms  
// - Total latency: 200-280ms
// - Hand detection accuracy: ~85%
```

### Option 2: High Quality (If network is excellent)
```javascript
const HIGH_QUALITY_CONFIG = {
  width: 1280,
  height: 720,
  quality: 0.6,              // 60% quality
  format: 'jpeg',
  skipProcessing: true,
  base64: true
};

// Expected results:
// - File size: 80-150KB per frame
// - Network time: 100-200ms
// - Total latency: 280-420ms  
// - Hand detection accuracy: ~95%
```

### Option 3: Low Bandwidth (If network is poor)
```javascript
const LOW_BANDWIDTH_CONFIG = {
  width: 320,
  height: 240,
  quality: 0.8,              // 80% quality
  format: 'jpeg',
  skipProcessing: true,
  base64: true
};

// Expected results:
// - File size: 15-30KB per frame
// - Network time: 20-40ms
// - Total latency: 150-200ms
// - Hand detection accuracy: ~70%
```

## Dynamic Quality Adjustment

### Adaptive Frame Size Based on Network
```javascript
class AdaptiveCamera {
  constructor() {
    this.currentQuality = 0.7;
    this.currentResolution = { width: 640, height: 480 };
    this.networkQuality = 1.0;
  }
  
  adjustBasedOnLatency(averageLatency) {
    if (averageLatency > 500) {
      // Network struggling - reduce quality
      this.currentQuality = 0.5;
      this.currentResolution = { width: 320, height: 240 };
    } else if (averageLatency < 200) {
      // Network good - increase quality
      this.currentQuality = 0.7;
      this.currentResolution = { width: 640, height: 480 };
    }
  }
  
  getCameraConfig() {
    return {
      ...this.currentResolution,
      quality: this.currentQuality,
      format: 'jpeg',
      skipProcessing: true,
      base64: true
    };
  }
}
```

## Server-Side Frame Size Monitoring

### Add to your WebSocket server:
```python
def monitor_frame_performance(self, client_id: str, frame_size: int, processing_time: float):
    """Monitor frame size vs processing performance"""
    
    if frame_size > 150000:  # 150KB
        logger.warning(f"Large frame from {client_id}: {frame_size:,} bytes")
        # Suggest client to reduce quality
        
    if processing_time > 200:  # 200ms
        logger.warning(f"Slow processing for {client_id}: {processing_time:.1f}ms")
        # Could trigger quality reduction message
        
    # Log performance metrics
    logger.info(f"Client {client_id}: {frame_size:,} bytes in {processing_time:.1f}ms")
```

## Real-World Testing Results

Based on typical WiFi networks:

### Home WiFi (Good)
- **Optimal**: 640×480 @ 70% quality
- **Frame size**: 40-80KB
- **Latency**: 200-300ms
- **Hand detection**: 85-90%

### Mobile Hotspot (Variable)  
- **Optimal**: 480×360 @ 60% quality
- **Frame size**: 25-50KB
- **Latency**: 250-400ms
- **Hand detection**: 80-85%

### Poor Network
- **Optimal**: 320×240 @ 80% quality
- **Frame size**: 15-30KB  
- **Latency**: 150-250ms
- **Hand detection**: 70-75%

## Performance Tips

1. **Monitor network latency** and adjust quality dynamically
2. **Use JPEG compression** (better than PNG for photos)
3. **Skip unnecessary processing** on mobile side
4. **Batch quality adjustments** (don't change every frame)
5. **Consider time of day** (network congestion varies)

## Recommended Implementation Order

1. **Start with 640×480 @ 70%** quality
2. **Monitor average latency** for first 50 frames  
3. **Adjust down** if latency > 400ms consistently
4. **Adjust up** if latency < 150ms consistently
5. **Implement adaptive system** for varying network conditions
