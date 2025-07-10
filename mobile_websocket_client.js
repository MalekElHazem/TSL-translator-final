// Simple React Native WebSocket Client for TSL
// Copy this into your React Native project

import React, { useState, useRef } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Alert } from 'react-native';

const TSLWebSocketClient = () => {
  const [connected, setConnected] = useState(false);
  const [status, setStatus] = useState('Disconnected');
  const [prediction, setPrediction] = useState(null);
  const wsRef = useRef(null);

  // Change this to your computer's IP address
  const SERVER_URL = 'ws://192.168.1.11:8765';

  const connect = () => {
    try {
      setStatus('Connecting...');
      wsRef.current = new WebSocket(SERVER_URL);

      wsRef.current.onopen = () => {
        setConnected(true);
        setStatus('Connected ✅');
        console.log('WebSocket connected');
      };

      wsRef.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log('Received:', data);
        
        // Handle different message types
        if (data.status === 'prediction') {
          setPrediction(data);
          setStatus(`Detected: ${data.predicted_class} (${(data.confidence * 100).toFixed(1)}%)`);
        } else if (data.status === 'waiting') {
          setStatus(`Collecting frames... ${data.frames_collected}/${data.frames_needed}`);
        } else if (data.type === 'pong') {
          setStatus('Server responded to ping ✅');
        }
      };

      wsRef.current.onclose = () => {
        setConnected(false);
        setStatus('Disconnected ❌');
        console.log('WebSocket disconnected');
      };

      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setStatus('Connection failed ❌');
        Alert.alert('Connection Error', 'Failed to connect to server');
      };

    } catch (error) {
      console.error('Connection error:', error);
      Alert.alert('Error', 'Failed to create WebSocket connection');
    }
  };

  const disconnect = () => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setConnected(false);
    setStatus('Disconnected');
    setPrediction(null);
  };

  const sendPing = () => {
    if (connected && wsRef.current) {
      wsRef.current.send(JSON.stringify({
        type: 'ping',
        timestamp: Date.now()
      }));
      setStatus('Ping sent...');
    } else {
      Alert.alert('Error', 'Not connected to server');
    }
  };

  const sendTestFrame = () => {
    if (connected && wsRef.current) {
      // Send a dummy base64 image for testing
      const dummyFrame = '/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/2wBDAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwA/8A8A';
      
      wsRef.current.send(JSON.stringify({
        type: 'frame',
        data: dummyFrame,
        timestamp: Date.now()
      }));
      setStatus('Test frame sent...');
    }
  };

  const adjustSensitivity = (factor) => {
    if (connected && wsRef.current) {
      wsRef.current.send(JSON.stringify({
        type: 'adjust_threshold',
        factor: factor
      }));
      setStatus(`Sensitivity adjusted (${factor > 1 ? 'increased' : 'decreased'})`);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>TSL WebSocket Test</Text>
      
      <View style={[styles.statusBar, {
        backgroundColor: connected ? '#d4edda' : '#f8d7da'
      }]}>
        <Text style={[styles.statusText, {
          color: connected ? '#155724' : '#721c24'
        }]}>
          {status}
        </Text>
      </View>

      {prediction && (
        <View style={styles.predictionBox}>
          <Text style={styles.predictionText}>
            Sign: {prediction.predicted_class}
          </Text>
          <Text style={styles.confidenceText}>
            Confidence: {(prediction.confidence * 100).toFixed(1)}%
          </Text>
        </View>
      )}

      <View style={styles.buttonContainer}>
        <TouchableOpacity 
          style={[styles.button, {backgroundColor: connected ? '#dc3545' : '#007bff'}]}
          onPress={connected ? disconnect : connect}
        >
          <Text style={styles.buttonText}>
            {connected ? 'Disconnect' : 'Connect'}
          </Text>
        </TouchableOpacity>

        <TouchableOpacity 
          style={[styles.button, {backgroundColor: '#28a745', opacity: connected ? 1 : 0.5}]}
          onPress={sendPing}
          disabled={!connected}
        >
          <Text style={styles.buttonText}>Test Ping</Text>
        </TouchableOpacity>

        <TouchableOpacity 
          style={[styles.button, {backgroundColor: '#17a2b8', opacity: connected ? 1 : 0.5}]}
          onPress={sendTestFrame}
          disabled={!connected}
        >
          <Text style={styles.buttonText}>Send Test Frame</Text>
        </TouchableOpacity>
      </View>

      <View style={styles.sensitivityContainer}>
        <Text style={styles.sensitivityTitle}>Motion Sensitivity:</Text>
        <View style={styles.sensitivityButtons}>
          <TouchableOpacity 
            style={[styles.smallButton, {opacity: connected ? 1 : 0.5}]}
            onPress={() => adjustSensitivity(0.8)}
            disabled={!connected}
          >
            <Text style={styles.smallButtonText}>Less</Text>
          </TouchableOpacity>
          <TouchableOpacity 
            style={[styles.smallButton, {opacity: connected ? 1 : 0.5}]}
            onPress={() => adjustSensitivity(1.2)}
            disabled={!connected}
          >
            <Text style={styles.smallButtonText}>More</Text>
          </TouchableOpacity>
        </View>
      </View>

      <Text style={styles.infoText}>
        Server: {SERVER_URL}
      </Text>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    backgroundColor: '#f8f9fa',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 20,
    color: '#333',
  },
  statusBar: {
    padding: 15,
    borderRadius: 8,
    marginBottom: 20,
    alignItems: 'center',
  },
  statusText: {
    fontSize: 16,
    fontWeight: 'bold',
  },
  predictionBox: {
    backgroundColor: '#e7f3ff',
    padding: 15,
    borderRadius: 8,
    marginBottom: 20,
    alignItems: 'center',
  },
  predictionText: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
  },
  confidenceText: {
    fontSize: 16,
    color: '#666',
    marginTop: 5,
  },
  buttonContainer: {
    marginBottom: 20,
  },
  button: {
    padding: 15,
    borderRadius: 8,
    alignItems: 'center',
    marginBottom: 10,
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
  sensitivityContainer: {
    alignItems: 'center',
    marginBottom: 20,
  },
  sensitivityTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 10,
    color: '#333',
  },
  sensitivityButtons: {
    flexDirection: 'row',
    gap: 10,
  },
  smallButton: {
    backgroundColor: '#6c757d',
    padding: 10,
    borderRadius: 5,
    minWidth: 60,
  },
  smallButtonText: {
    color: 'white',
    fontSize: 14,
    textAlign: 'center',
  },
  infoText: {
    fontSize: 12,
    color: '#666',
    textAlign: 'center',
  },
});

export default TSLWebSocketClient;
