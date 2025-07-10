// App.js
// Main Expo app component for TSL WebSocket detection

import React, { useState, useRef, useEffect } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  Alert,
  Dimensions,
  Platform,
  StatusBar,
  ScrollView
} from 'react-native';
import { Camera } from 'expo-camera';
import * as FileSystem from 'expo-file-system';
import TSLWebSocketService from './services/TSLWebSocketService';

const { width, height } = Dimensions.get('window');

export default function App() {
  // State management
  const [hasPermission, setHasPermission] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isDetecting, setIsDetecting] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [status, setStatus] = useState('Disconnected');
  const [frameCount, setFrameCount] = useState(0);
  const [serverStats, setServerStats] = useState({
    framesCollected: 0,
    framesNeeded: 16
  });

  // Refs
  const cameraRef = useRef(null);
  const wsService = useRef(new TSLWebSocketService());
  const captureInterval = useRef(null);

  // Configuration
  const SERVER_URL = 'ws://192.168.1.11:8765'; // Change to your server IP
  const CAPTURE_INTERVAL = 200; // 200ms = 5 FPS

  useEffect(() => {
    initializeApp();
    return () => cleanup();
  }, []);

  const initializeApp = async () => {
    // Request camera permissions
    const { status } = await Camera.requestCameraPermissionsAsync();
    setHasPermission(status === 'granted');

    if (status !== 'granted') {
      Alert.alert('Permission Required', 'Camera permission is needed for sign language detection');
      return;
    }

    // Setup WebSocket handlers
    setupWebSocket();
  };

  const setupWebSocket = () => {
    const ws = wsService.current;

    ws.setConnectionHandler((connected, message) => {
      setIsConnected(connected);
      setStatus(message);
    });

    ws.setMessageHandler((data) => {
      handleServerMessage(data);
    });

    ws.setErrorHandler((error) => {
      console.error('WebSocket error:', error);
      Alert.alert('Connection Error', 'Failed to connect to server');
    });
  };

  const handleServerMessage = (data) => {
    console.log('📥 Server message:', data);

    switch (data.status) {
      case 'waiting':
        setServerStats({
          framesCollected: data.frames_collected || 0,
          framesNeeded: data.frames_needed || 16
        });
        setStatus(`Collecting frames... ${data.frames_collected}/${data.frames_needed}`);
        break;

      case 'collecting':
        setServerStats({
          framesCollected: data.frames_collected || 0,
          framesNeeded: data.frames_needed || 16
        });
        setStatus(`Processing... ${data.frames_collected}/${data.frames_needed}`);
        break;

      case 'prediction':
        setPrediction(data);
        const confidenceText = data.above_threshold 
          ? `${data.predicted_class} (${(data.confidence * 100).toFixed(1)}%)`
          : `Low confidence: ${data.predicted_class} (${(data.confidence * 100).toFixed(1)}%)`;
        setStatus(`🎯 ${confidenceText}`);
        break;

      case 'error':
        setStatus(`❌ Error: ${data.message}`);
        break;

      case 'no_motion':
        setStatus('👋 Waiting for hand movement...');
        break;

      default:
        if (data.type === 'pong') {
          setStatus('✅ Server ping successful');
        } else if (data.type === 'threshold_updated') {
          setStatus(`⚙️ Sensitivity updated: ${data.new_threshold.toFixed(4)}`);
        }
        break;
    }
  };

  const connect = () => {
    setStatus('🔄 Connecting...');
    wsService.current.connect(SERVER_URL);
  };

  const disconnect = () => {
    stopDetection();
    wsService.current.disconnect();
    setPrediction(null);
    setFrameCount(0);
  };

  const startDetection = () => {
    if (!isConnected) {
      Alert.alert('Error', 'Please connect to server first');
      return;
    }

    if (!cameraRef.current) {
      Alert.alert('Error', 'Camera not ready');
      return;
    }

    setIsDetecting(true);
    setStatus('🎯 Starting detection...');
    setFrameCount(0);

    // Start capturing frames
    captureInterval.current = setInterval(async () => {
      await captureAndSendFrame();
    }, CAPTURE_INTERVAL);
  };

  const stopDetection = () => {
    if (captureInterval.current) {
      clearInterval(captureInterval.current);
      captureInterval.current = null;
    }

    setIsDetecting(false);
    setStatus(isConnected ? 'Connected - Ready to detect' : 'Disconnected');
    setPrediction(null);
    setFrameCount(0);
  };

  const captureAndSendFrame = async () => {
    try {
      if (!cameraRef.current || !isConnected) return;

      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.7,
        base64: true,
        skipProcessing: true,
      });

      // Send base64 image to server
      wsService.current.sendFrame(photo.base64);
      setFrameCount(prev => prev + 1);

    } catch (error) {
      console.error('Frame capture error:', error);
    }
  };

  const sendPing = () => {
    if (isConnected) {
      wsService.current.sendPing();
      setStatus('📡 Ping sent...');
    } else {
      Alert.alert('Error', 'Not connected to server');
    }
  };

  const adjustSensitivity = (factor) => {
    if (isConnected) {
      wsService.current.adjustThreshold(factor);
      setStatus(`⚙️ Adjusting sensitivity...`);
    } else {
      Alert.alert('Error', 'Not connected to server');
    }
  };

  const cleanup = () => {
    stopDetection();
    wsService.current.disconnect();
  };

  // Handle permission denial
  if (hasPermission === null) {
    return (
      <View style={styles.container}>
        <Text style={styles.loadingText}>Requesting camera permission...</Text>
      </View>
    );
  }

  if (hasPermission === false) {
    return (
      <View style={styles.container}>
        <Text style={styles.errorText}>Camera permission denied</Text>
        <TouchableOpacity style={styles.button} onPress={initializeApp}>
          <Text style={styles.buttonText}>Request Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" />
      
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <Text style={styles.title}>🤟 TSL Real-Time Translator</Text>

        {/* Connection Status */}
        <View style={[styles.statusContainer, {
          backgroundColor: isConnected ? '#d4edda' : '#f8d7da'
        }]}>
          <Text style={[styles.statusText, {
            color: isConnected ? '#155724' : '#721c24'
          }]}>
            {status}
          </Text>
        </View>

        {/* Camera View */}
        <View style={styles.cameraContainer}>
          <Camera
            ref={cameraRef}
            style={styles.camera}
            type={Camera.Constants.Type.front}
            ratio="16:9"
          />
          
          {/* Frame Count Overlay */}
          <View style={styles.overlay}>
            <Text style={styles.overlayText}>Frames: {frameCount}</Text>
            <Text style={styles.overlayText}>
              Buffer: {serverStats.framesCollected}/{serverStats.framesNeeded}
            </Text>
          </View>
        </View>

        {/* Prediction Display */}
        {prediction && (
          <View style={[styles.predictionContainer, {
            backgroundColor: prediction.above_threshold ? '#d4edda' : '#fff3cd'
          }]}>
            <Text style={styles.predictionText}>
              {prediction.predicted_class}
            </Text>
            <Text style={styles.confidenceText}>
              {(prediction.confidence * 100).toFixed(1)}% confidence
            </Text>
            {prediction.above_threshold ? (
              <Text style={styles.resultText}>✅ High Confidence</Text>
            ) : (
              <Text style={styles.resultText}>⚠️ Low Confidence</Text>
            )}
          </View>
        )}

        {/* Control Buttons */}
        <View style={styles.controlsContainer}>
          <View style={styles.buttonRow}>
            <TouchableOpacity 
              style={[styles.button, { 
                backgroundColor: isConnected ? '#dc3545' : '#007bff' 
              }]}
              onPress={isConnected ? disconnect : connect}
            >
              <Text style={styles.buttonText}>
                {isConnected ? '🔌 Disconnect' : '🔗 Connect'}
              </Text>
            </TouchableOpacity>

            <TouchableOpacity 
              style={[styles.button, { 
                backgroundColor: '#17a2b8',
                opacity: isConnected ? 1 : 0.5
              }]}
              onPress={sendPing}
              disabled={!isConnected}
            >
              <Text style={styles.buttonText}>📡 Ping</Text>
            </TouchableOpacity>
          </View>

          <View style={styles.buttonRow}>
            <TouchableOpacity 
              style={[styles.button, { 
                backgroundColor: isDetecting ? '#ffc107' : '#28a745',
                opacity: isConnected ? 1 : 0.5
              }]}
              onPress={isDetecting ? stopDetection : startDetection}
              disabled={!isConnected}
            >
              <Text style={[styles.buttonText, {
                color: isDetecting ? '#212529' : 'white'
              }]}>
                {isDetecting ? '⏹️ Stop Detection' : '🎯 Start Detection'}
              </Text>
            </TouchableOpacity>
          </View>

          {/* Sensitivity Controls */}
          <View style={styles.sensitivityContainer}>
            <Text style={styles.sensitivityTitle}>Motion Sensitivity:</Text>
            <View style={styles.buttonRow}>
              <TouchableOpacity 
                style={[styles.smallButton, {opacity: isConnected ? 1 : 0.5}]}
                onPress={() => adjustSensitivity(0.8)}
                disabled={!isConnected}
              >
                <Text style={styles.smallButtonText}>Less</Text>
              </TouchableOpacity>
              <TouchableOpacity 
                style={[styles.smallButton, {opacity: isConnected ? 1 : 0.5}]}
                onPress={() => adjustSensitivity(1.2)}
                disabled={!isConnected}
              >
                <Text style={styles.smallButtonText}>More</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>

        <Text style={styles.serverInfo}>
          Server: {SERVER_URL}
        </Text>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8f9fa',
    paddingTop: Platform.OS === 'ios' ? 50 : StatusBar.currentHeight + 10,
  },
  scrollContent: {
    padding: 20,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 20,
    color: '#333',
  },
  statusContainer: {
    padding: 15,
    borderRadius: 10,
    marginBottom: 20,
    alignItems: 'center',
  },
  statusText: {
    fontSize: 16,
    fontWeight: 'bold',
    textAlign: 'center',
  },
  cameraContainer: {
    height: 300,
    borderRadius: 15,
    overflow: 'hidden',
    marginBottom: 20,
    position: 'relative',
  },
  camera: {
    flex: 1,
  },
  overlay: {
    position: 'absolute',
    top: 10,
    right: 10,
    backgroundColor: 'rgba(0,0,0,0.7)',
    padding: 8,
    borderRadius: 5,
  },
  overlayText: {
    color: 'white',
    fontSize: 12,
    fontWeight: 'bold',
  },
  predictionContainer: {
    padding: 20,
    borderRadius: 15,
    marginBottom: 20,
    alignItems: 'center',
  },
  predictionText: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 5,
  },
  confidenceText: {
    fontSize: 18,
    color: '#666',
    marginBottom: 10,
  },
  resultText: {
    fontSize: 16,
    fontWeight: 'bold',
  },
  controlsContainer: {
    marginBottom: 20,
  },
  buttonRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 10,
  },
  button: {
    flex: 1,
    padding: 15,
    borderRadius: 10,
    alignItems: 'center',
    marginHorizontal: 5,
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
  sensitivityContainer: {
    alignItems: 'center',
    marginTop: 20,
  },
  sensitivityTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 10,
  },
  smallButton: {
    backgroundColor: '#6c757d',
    padding: 12,
    borderRadius: 8,
    minWidth: 80,
    marginHorizontal: 10,
  },
  smallButtonText: {
    color: 'white',
    fontSize: 14,
    fontWeight: 'bold',
    textAlign: 'center',
  },
  serverInfo: {
    fontSize: 12,
    color: '#666',
    textAlign: 'center',
    marginTop: 10,
  },
  loadingText: {
    fontSize: 18,
    textAlign: 'center',
    color: '#666',
  },
  errorText: {
    fontSize: 18,
    textAlign: 'center',
    color: '#dc3545',
    marginBottom: 20,
  },
});
