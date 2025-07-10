// services/WebSocketService.js
// WebSocket service specifically designed for Expo Go

export class TSLWebSocketService {
  constructor() {
    this.ws = null;
    this.isConnected = false;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.reconnectDelay = 2000;
    this.messageQueue = [];
    
    // Event handlers
    this.onConnectionChange = null;
    this.onMessage = null;
    this.onError = null;
  }

  // Connect to your WebSocket server
  connect(serverUrl = 'ws://192.168.1.11:8765') {
    console.log('🔄 Connecting to:', serverUrl);
    
    try {
      this.ws = new WebSocket(serverUrl);
      
      this.ws.onopen = (event) => {
        console.log('✅ WebSocket connected');
        this.isConnected = true;
        this.reconnectAttempts = 0;
        this.flushMessageQueue();
        
        if (this.onConnectionChange) {
          this.onConnectionChange(true, 'Connected to server');
        }
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          console.log('📥 Received:', data);
          
          if (this.onMessage) {
            this.onMessage(data);
          }
        } catch (error) {
          console.error('Error parsing message:', error);
        }
      };

      this.ws.onclose = (event) => {
        console.log('🔌 WebSocket closed:', event.code);
        this.isConnected = false;
        
        if (this.onConnectionChange) {
          this.onConnectionChange(false, 'Disconnected from server');
        }
        
        // Auto-reconnect if not intentionally closed
        if (event.code !== 1000 && this.reconnectAttempts < this.maxReconnectAttempts) {
          this.scheduleReconnect(serverUrl);
        }
      };

      this.ws.onerror = (error) => {
        console.error('❌ WebSocket error:', error);
        
        if (this.onError) {
          this.onError(error);
        }
      };

    } catch (error) {
      console.error('Failed to create WebSocket:', error);
      if (this.onError) {
        this.onError(error);
      }
    }
  }

  scheduleReconnect(serverUrl) {
    this.reconnectAttempts++;
    console.log(`🔄 Reconnect attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts}`);
    
    setTimeout(() => {
      if (!this.isConnected) {
        this.connect(serverUrl);
      }
    }, this.reconnectDelay);
  }

  send(message) {
    if (this.isConnected && this.ws.readyState === WebSocket.OPEN) {
      try {
        const messageStr = JSON.stringify(message);
        this.ws.send(messageStr);
        console.log('📤 Sent:', message.type);
      } catch (error) {
        console.error('Error sending message:', error);
        this.messageQueue.push(message);
      }
    } else {
      console.log('📋 Queued message:', message.type);
      this.messageQueue.push(message);
    }
  }

  sendFrame(base64Image) {
    this.send({
      type: 'frame',
      data: base64Image,
      timestamp: Date.now()
    });
  }

  sendPing() {
    this.send({
      type: 'ping',
      timestamp: Date.now()
    });
  }

  adjustThreshold(factor) {
    this.send({
      type: 'adjust_threshold',
      factor: factor
    });
  }

  flushMessageQueue() {
    while (this.messageQueue.length > 0) {
      const message = this.messageQueue.shift();
      this.send(message);
    }
  }

  disconnect() {
    console.log('🔌 Disconnecting WebSocket');
    this.reconnectAttempts = this.maxReconnectAttempts; // Prevent reconnection
    
    if (this.ws) {
      this.ws.close(1000, 'User disconnected');
      this.ws = null;
    }
    
    this.isConnected = false;
    this.messageQueue = [];
  }

  // Set event handlers
  setConnectionHandler(handler) {
    this.onConnectionChange = handler;
  }

  setMessageHandler(handler) {
    this.onMessage = handler;
  }

  setErrorHandler(handler) {
    this.onError = handler;
  }
}

export default TSLWebSocketService;
