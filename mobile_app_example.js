/**
 * Mobile App Example for Real-Time Sign Language Detection
 * This is a React Native component that connects to your WebSocket server
 * 
 * Installation:
 * npm install react-native-camera
 * npm install react-native-fs
 */

import React, { useState, useRef, useEffect } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Alert } from 'react-native';
import { RNCamera } from 'react-native-camera';

const SignLanguageDetector = () => {
  const [isConnected, setIsConnected] = useState(false);
  const [isDetecting, setIsDetecting] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [status, setStatus] = useState('Disconnected');
  const [motionInfo, setMotionInfo] = useState('');
  
  const cameraRef = useRef(null);
  const socketRef = useRef(null);
  const intervalRef = useRef(null);
  
  // Replace with your local server IP
  const SERVER_URL = 'ws://192.168.1.100:8765'; // Change to your computer's IP
  
  useEffect(() => {
    return () => {
      disconnect();
    };
  }, []);
  
  const connect = () => {
    try {
      const socket = new WebSocket(SERVER_URL);
      socketRef.current = socket;
      
      socket.onopen = () => {
        setIsConnected(true);
        setStatus('Connected to server');
        console.log('Connected to WebSocket server');
      };
      
      socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleServerResponse(data);
      };
      
      socket.onclose = () => {
        setIsConnected(false);
        setIsDetecting(false);
        setStatus('Disconnected from server');
        console.log('Disconnected from WebSocket server');
      };
      
      socket.onerror = (error) => {
        console.error('WebSocket error:', error);
        setStatus('Connection error');
        Alert.alert('Connection Error', 'Failed to connect to server');
      };
      
    } catch (error) {
      console.error('Connection error:', error);
      Alert.alert('Error', 'Failed to connect to server');
    }
  };
  
  const disconnect = () => {
    if (socketRef.current) {
      socketRef.current.close();
      socketRef.current = null;
    }
    
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    
    setIsConnected(false);
    setIsDetecting(false);
    setStatus('Disconnected');
  };
  
  const startDetection = () => {
    if (!isConnected || !cameraRef.current) return;
    
    setIsDetecting(true);
    setStatus('Starting detection...');
    
    // Capture and send frames every 100ms (10 FPS)
    intervalRef.current = setInterval(async () => {
      if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
        await captureAndSendFrame();
      }
    }, 100);
  };
  
  const stopDetection = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    
    setIsDetecting(false);
    setPrediction(null);
    setStatus('Detection stopped');
  };
  
  const captureAndSendFrame = async () => {
    try {
      if (!cameraRef.current) return;
      
      const options = {
        quality: 0.7,
        base64: true,
        width: 640,
        height: 480,
        skipProcessing: true,
      };
      
      const data = await cameraRef.current.takePictureAsync(options);
      
      if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
        socketRef.current.send(JSON.stringify({
          type: 'frame',
          data: data.base64
        }));
      }
      
    } catch (error) {
      console.error('Error capturing frame:', error);
    }
  };
  
  const handleServerResponse = (data) => {
    switch (data.status) {
      case 'waiting':
        setStatus(`Collecting frames... (${data.frames_collected}/${data.frames_needed})`);
        setMotionInfo(`Motion: ${data.motion_detected ? 'Detected' : 'Not detected'}`);
        break;
        
      case 'collecting':
        setStatus(`Collecting frames... (${data.frames_collected}/${data.frames_needed})`);
        break;
        
      case 'prediction':
        setPrediction(data);
        setStatus(data.above_threshold ? 
          `Prediction: ${data.predicted_class} (${(data.confidence * 100).toFixed(1)}%)` :
          `Low confidence: ${data.predicted_class} (${(data.confidence * 100).toFixed(1)}%)`
        );
        setMotionInfo(`Motion: ${data.motion_score.toFixed(6)}`);
        break;
        
      case 'error':
        setStatus('Error: ' + data.message);
        break;
    }
  };
  
  const adjustThreshold = (factor) => {
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify({
        type: 'adjust_threshold',
        factor: factor
      }));
    }
  };
  
  return (
    <View style={styles.container}>
      <Text style={styles.title}>🤟 Sign Language Detector</Text>
      
      <View style={styles.cameraContainer}>
        <RNCamera
          ref={cameraRef}
          style={styles.camera}
          type={RNCamera.Constants.Type.front}
          flashMode={RNCamera.Constants.FlashMode.off}
          androidCameraPermissionOptions={{
            title: 'Permission to use camera',
            message: 'We need your permission to use your camera',
            buttonPositive: 'Ok',
            buttonNegative: 'Cancel',
          }}
        />
      </View>
      
      <View style={styles.statusContainer}>
        <Text style={[styles.status, { color: isConnected ? '#28a745' : '#dc3545' }]}>
          {status}
        </Text>
        {motionInfo && <Text style={styles.motionInfo}>{motionInfo}</Text>}
      </View>
      
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
          
          {prediction.top_predictions && (
            <View style={styles.topPredictions}>
              <Text style={styles.topPredictionsTitle}>Top Predictions:</Text>
              {prediction.top_predictions.slice(0, 3).map((pred, index) => (
                <Text key={index} style={styles.topPredictionItem}>
                  {pred.class}: {(pred.confidence * 100).toFixed(1)}%
                </Text>
              ))}
            </View>
          )}
        </View>
      )}
      
      <View style={styles.buttonContainer}>
        <TouchableOpacity 
          style={[styles.button, { backgroundColor: isConnected ? '#dc3545' : '#007bff' }]}
          onPress={isConnected ? disconnect : connect}
        >
          <Text style={styles.buttonText}>
            {isConnected ? 'Disconnect' : 'Connect'}
          </Text>
        </TouchableOpacity>
        
        <TouchableOpacity 
          style={[styles.button, { 
            backgroundColor: isDetecting ? '#ffc107' : '#28a745',
            opacity: isConnected ? 1 : 0.5
          }]}
          onPress={isDetecting ? stopDetection : startDetection}
          disabled={!isConnected}
        >
          <Text style={styles.buttonText}>
            {isDetecting ? 'Stop Detection' : 'Start Detection'}
          </Text>
        </TouchableOpacity>
      </View>
      
      <View style={styles.thresholdContainer}>
        <Text style={styles.thresholdTitle}>Motion Threshold:</Text>
        <View style={styles.thresholdButtons}>
          <TouchableOpacity 
            style={styles.thresholdButton}
            onPress={() => adjustThreshold(0.8)}
            disabled={!isConnected}
          >
            <Text style={styles.thresholdButtonText}>-</Text>
          </TouchableOpacity>
          <TouchableOpacity 
            style={styles.thresholdButton}
            onPress={() => adjustThreshold(1.2)}
            disabled={!isConnected}
          >
            <Text style={styles.thresholdButtonText}>+</Text>
          </TouchableOpacity>
        </View>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8f9fa',
    padding: 20,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 20,
    color: '#333',
  },
  cameraContainer: {
    flex: 1,
    borderRadius: 10,
    overflow: 'hidden',
    marginBottom: 20,
  },
  camera: {
    flex: 1,
  },
  statusContainer: {
    marginBottom: 15,
  },
  status: {
    fontSize: 16,
    fontWeight: 'bold',
    textAlign: 'center',
  },
  motionInfo: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
    marginTop: 5,
  },
  predictionContainer: {
    padding: 15,
    borderRadius: 10,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: '#dee2e6',
  },
  predictionText: {
    fontSize: 20,
    fontWeight: 'bold',
    textAlign: 'center',
    color: '#333',
  },
  confidenceText: {
    fontSize: 16,
    textAlign: 'center',
    color: '#666',
    marginTop: 5,
  },
  topPredictions: {
    marginTop: 10,
  },
  topPredictionsTitle: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 5,
  },
  topPredictionItem: {
    fontSize: 12,
    color: '#666',
    marginBottom: 2,
  },
  buttonContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginBottom: 20,
  },
  button: {
    padding: 15,
    borderRadius: 8,
    minWidth: 120,
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
    textAlign: 'center',
  },
  thresholdContainer: {
    alignItems: 'center',
  },
  thresholdTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 10,
  },
  thresholdButtons: {
    flexDirection: 'row',
    gap: 10,
  },
  thresholdButton: {
    backgroundColor: '#6c757d',
    padding: 10,
    borderRadius: 5,
    minWidth: 40,
  },
  thresholdButtonText: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
    textAlign: 'center',
  },
});

export default SignLanguageDetector;
