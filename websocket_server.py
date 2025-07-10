"""
Real-time Sign Language Detection WebSocket Server
Handles live video streaming from mobile apps for real-time predictions
"""
import asyncio
import websockets
import json
import base64
import cv2
import numpy as np
import torch
from collections import deque
import time
import logging
from typing import Optional, Dict, Any
import mediapipe as mp
import os

from models import SignLanguageModel
from configs import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# WebSocket debug frames directory
WEBSOCKET_DEBUG_ROOT_DIR = "websocket_debug_frames"
os.makedirs(WEBSOCKET_DEBUG_ROOT_DIR, exist_ok=True)

class OptimizedRealTimeDetector:
    def __init__(self):
        self.model = None
        self.class_names = None
        self.device = None
        self.transform = None
        self.hands_detector = None
        
        # Frame buffers for each client
        self.client_buffers: Dict[str, deque] = {}
        self.client_histories: Dict[str, deque] = {}
        self.client_motion_histories: Dict[str, deque] = {}
        self.client_prev_frames: Dict[str, Optional[np.ndarray]] = {}
        
        # Thresholds
        self.motion_threshold = config.MOTION_THRESHOLD
        self.confidence_threshold = config.CONFIDENCE_THRESHOLD
        
        # Add adaptive frame rate
        self.adaptive_fps = {}  # Per client FPS adjustment
        self.connection_quality = {}  # Track connection quality
        
        # Add reconnection handling
        self.client_sessions = {}  # Persistent client sessions
        
        # WebSocket frame debugging
        self.client_debug_dirs = {}  # Store debug directories for each client
        self.client_frame_counters = {}  # Frame counters for each client
        
        self._initialize_model()
        self._initialize_mediapipe()
        self._initialize_transforms()
    
    def _initialize_model(self):
        """Initialize the sign language model"""
        try:
            # Load class names
            with open(config.CLASS_NAMES_FILE, 'r') as f:
                self.class_names = [line.strip() for line in f.readlines()]
            
            num_classes = len(self.class_names)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Load model
            self.model = SignLanguageModel(
                num_classes=num_classes,
                input_size=config.INPUT_SIZE,
                hidden_size=config.HIDDEN_SIZE,
                dropout_rate=config.DROPOUT_RATE,
                bidirectional=config.BIDIRECTIONAL,
                num_lstm_layers=config.NUM_LSTM_LAYERS
            ).to(self.device)
            
            if hasattr(torch, 'load') and 'weights_only' in torch.load.__code__.co_varnames:
                self.model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=self.device, weights_only=True))
            else:
                self.model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=self.device))
            
            self.model.eval()
            logger.info(f"Model loaded successfully. Device: {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    def _initialize_mediapipe(self):
        """Initialize MediaPipe hands detector"""
        try:
            mp_hands = mp.solutions.hands
            self.hands_detector = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.5
            )
            logger.info("MediaPipe hands detector initialized")
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe: {e}")
            raise
    
    def _initialize_transforms(self):
        """Initialize image transforms"""
        from torchvision import transforms
        
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE), 
                            interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.ToTensor(),
            normalize
        ])
        logger.info("Image transforms initialized")
    
    def _apply_mediapipe_mask_and_grayscale(self, image_rgb: np.ndarray) -> tuple:
        """Apply MediaPipe masking and convert to grayscale"""
        image_rgb.flags.writeable = False
        results = self.hands_detector.process(image_rgb)
        image_rgb.flags.writeable = True
        
        mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
        hands_detected = False
        
        if results.multi_hand_landmarks:
            hands_detected = True
            for hand_landmarks in results.multi_hand_landmarks:
                landmark_points = np.array([
                    (int(lm.x * image_rgb.shape[1]), int(lm.y * image_rgb.shape[0]))
                    for lm in hand_landmarks.landmark
                ], dtype=np.int32)
                
                # Clip coordinates to image bounds
                landmark_points[:, 0] = np.clip(landmark_points[:, 0], 0, image_rgb.shape[1] - 1)
                landmark_points[:, 1] = np.clip(landmark_points[:, 1], 0, image_rgb.shape[0] - 1)
                
                if len(landmark_points) >= 3:
                    try:
                        hull = cv2.convexHull(landmark_points)
                        cv2.fillConvexPoly(mask, hull, 255)
                    except Exception as e:
                        logger.warning(f"Hull failed: {e}")
                        for point in landmark_points:
                            cv2.circle(mask, tuple(point), 5, 255, -1)
        
        gray_source = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        masked_gray_image = cv2.bitwise_and(gray_source, gray_source, mask=mask)
        
        # Add debug logging for first few frames
        total_mask_pixels = np.count_nonzero(mask)
        total_masked_pixels = np.count_nonzero(masked_gray_image)
        
        return masked_gray_image, mask
    
    def _calculate_motion(self, frame: np.ndarray, prev_frame: Optional[np.ndarray]) -> float:
        """Calculate motion score between frames"""
        if prev_frame is None:
            return 0.0
        
        if frame.shape != prev_frame.shape:
            return 0.0
        
        frame_diff = cv2.absdiff(frame, prev_frame)
        motion_score = np.sum(frame_diff) / (frame.shape[0] * frame.shape[1] * 255.0)
        return motion_score
    
    def _predict_sequence(self, frame_buffer: deque) -> Dict[str, Any]:
        """Make prediction on frame sequence"""
        if len(frame_buffer) < config.SEQUENCE_LENGTH:
            return {
                "status": "collecting",
                "frames_collected": len(frame_buffer),
                "frames_needed": config.SEQUENCE_LENGTH
            }
        
        try:
            # Stack frames for model input
            input_tensor = torch.stack(list(frame_buffer)).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                
                # Apply neutral handicap if configured
                neutral_idx = -1
                if 'neutral' in self.class_names:
                    neutral_idx = self.class_names.index('neutral')
                    if config.NEUTRAL_HANDICAP > 0:
                        probs[0, neutral_idx] = max(0.0, probs[0, neutral_idx] - config.NEUTRAL_HANDICAP)
                        probs = probs / probs.sum(dim=1, keepdim=True)
                
                top_prob, top_class_idx = torch.max(probs, 1)
                predicted_idx = top_class_idx.item()
                confidence = top_prob.item()
                probabilities = probs[0].cpu().numpy()
            
            # Get top 5 predictions
            sorted_indices = np.argsort(probabilities)[::-1][:5]
            top_predictions = [
                {
                    "class": self.class_names[idx],
                    "confidence": float(probabilities[idx])
                }
                for idx in sorted_indices
            ]
            
            return {
                "status": "prediction",
                "predicted_class": self.class_names[predicted_idx],
                "confidence": float(confidence),
                "top_predictions": top_predictions,
                "above_threshold": confidence > self.confidence_threshold
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def process_frame(self, client_id: str, frame_data: str) -> Dict[str, Any]:
        """Process a single frame from client"""
        try:
            # Monitor message size
            message_size = len(frame_data)
            if message_size > 500000:  # 500KB threshold
                logger.warning(f"Large message from {client_id}: {message_size:,} bytes")
            elif message_size > 1000000:  # 1MB threshold
                logger.error(f"Very large message from {client_id}: {message_size:,} bytes")
                
            # Initialize client buffers if needed
            if client_id not in self.client_buffers:
                self.client_buffers[client_id] = deque(maxlen=config.SEQUENCE_LENGTH)
                self.client_histories[client_id] = deque(maxlen=config.HISTORY_SIZE)
                self.client_motion_histories[client_id] = deque(maxlen=5)
                self.client_prev_frames[client_id] = None
            
            # Decode base64 image
            img_data = base64.b64decode(frame_data)
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return {"status": "error", "message": "Could not decode frame"}
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Calculate motion
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            motion_score = self._calculate_motion(frame_gray, self.client_prev_frames[client_id])
            self.client_motion_histories[client_id].append(motion_score)
            avg_motion = sum(self.client_motion_histories[client_id]) / len(self.client_motion_histories[client_id])
            self.client_prev_frames[client_id] = frame_gray
            
            # Apply MediaPipe masking
            processed_frame, mask = self._apply_mediapipe_mask_and_grayscale(frame_rgb)
            
            # Save the WebSocket frame for debugging (original + masked)
            self._save_websocket_frame(client_id, frame, processed_frame)
            
            # Transform frame
            transformed_frame = self.transform(processed_frame)
            
            # Ensure correct shape
            if len(transformed_frame.shape) == 2:
                transformed_frame = transformed_frame.unsqueeze(0)
            if transformed_frame.shape[0] != 1:
                transformed_frame = transformed_frame[0, :, :].unsqueeze(0)
            
            # Add to buffer
            self.client_buffers[client_id].append(transformed_frame)
            
            # Check if we should make a prediction
            should_predict = (len(self.client_buffers[client_id]) == config.SEQUENCE_LENGTH and 
                            avg_motion > self.motion_threshold)
            
            if should_predict:
                prediction_result = self._predict_sequence(self.client_buffers[client_id])
                prediction_result["motion_score"] = float(avg_motion)
                prediction_result["motion_threshold"] = float(self.motion_threshold)
                return prediction_result
            else:
                return {
                    "status": "waiting",
                    "frames_collected": len(self.client_buffers[client_id]),
                    "frames_needed": config.SEQUENCE_LENGTH,
                    "motion_score": float(avg_motion),
                    "motion_threshold": float(self.motion_threshold),
                    "motion_detected": avg_motion > self.motion_threshold
                }
                
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return {"status": "error", "message": str(e)}
    
    def cleanup_client(self, client_id: str):
        """Clean up client data when disconnected"""
        # Create session summary before cleanup
        self._create_session_summary(client_id)
        
        # Clean up all client data
        if client_id in self.client_buffers:
            del self.client_buffers[client_id]
        if client_id in self.client_histories:
            del self.client_histories[client_id]
        if client_id in self.client_motion_histories:
            del self.client_motion_histories[client_id]
        if client_id in self.client_prev_frames:
            del self.client_prev_frames[client_id]
        if client_id in self.adaptive_fps:
            del self.adaptive_fps[client_id]
        if client_id in self.connection_quality:
            del self.connection_quality[client_id]
        if client_id in self.client_sessions:
            del self.client_sessions[client_id]
        # Clean up debug directories tracking
        if client_id in self.client_debug_dirs:
            del self.client_debug_dirs[client_id]
        if client_id in self.client_frame_counters:
            del self.client_frame_counters[client_id]
        logger.info(f"Cleaned up client data for {client_id}")
    
    def _setup_client_debug_dir(self, client_id: str) -> str:
        """Setup debug directory for a client if not already created"""
        if client_id not in self.client_debug_dirs:
            # Create a unique directory for this client session
            timestamp = int(time.time())
            client_safe_id = client_id.replace(":", "_").replace(".", "_")
            debug_dir = os.path.join(WEBSOCKET_DEBUG_ROOT_DIR, f"client_{client_safe_id}_{timestamp}")
            os.makedirs(debug_dir, exist_ok=True)
            
            self.client_debug_dirs[client_id] = debug_dir
            self.client_frame_counters[client_id] = 0
            
            # Create session info file
            session_info = {
                "client_id": client_id,
                "session_start": timestamp,
                "session_start_readable": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
            }
            with open(os.path.join(debug_dir, "session_info.json"), "w") as f:
                json.dump(session_info, f, indent=2)
            
            logger.info(f"Created WebSocket debug directory: {debug_dir}")
        
        return self.client_debug_dirs[client_id]
    
    def _save_websocket_frame(self, client_id: str, frame: np.ndarray, processed_frame: Optional[np.ndarray] = None) -> None:
        """Save a frame from WebSocket to debug directory"""
        try:
            debug_dir = self._setup_client_debug_dir(client_id)
            
            # Increment frame counter
            self.client_frame_counters[client_id] += 1
            frame_num = self.client_frame_counters[client_id]
            
            # Save frame with timestamp and counter
            timestamp = int(time.time() * 1000)  # milliseconds
            
            # Save original frame (BGR format for OpenCV)
            original_frame_filename = f"frame_{frame_num:06d}_{timestamp}_original.jpg"
            original_frame_path = os.path.join(debug_dir, original_frame_filename)
            cv2.imwrite(original_frame_path, frame)
            
            # Save masked frame if provided
            if processed_frame is not None:
                masked_frame_filename = f"frame_{frame_num:06d}_{timestamp}_masked.jpg"
                masked_frame_path = os.path.join(debug_dir, masked_frame_filename)
                
                # Check if processed_frame is grayscale or has valid data
                if len(processed_frame.shape) == 2:  # Grayscale
                    # Convert grayscale to BGR for better visibility
                    processed_frame_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
                    cv2.imwrite(masked_frame_path, processed_frame_bgr)
                else:  # Color
                    cv2.imwrite(masked_frame_path, processed_frame)
                
                # Log frame info for debugging
                if frame_num <= 5:  # Log first 5 frames for debugging
                    original_size = frame.shape if frame is not None else "None"
                    masked_size = processed_frame.shape if processed_frame is not None else "None"
                    masked_nonzero = np.count_nonzero(processed_frame) if processed_frame is not None else 0
                    logger.info(f"Frame {frame_num} - Original: {original_size}, Masked: {masked_size}, Non-zero pixels: {masked_nonzero}")
            else:
                logger.warning(f"Frame {frame_num}: No processed frame provided for {client_id}")
            
            # Log every 10th frame to avoid spam
            if frame_num % 10 == 0:
                logger.info(f"Saved WebSocket frame {frame_num} (original + masked) for {client_id}")
                
        except Exception as e:
            logger.error(f"Error saving WebSocket frame: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _create_session_summary(self, client_id: str) -> None:
        """Create a session summary similar to FastAPI's request_summary.txt"""
        try:
            if client_id not in self.client_debug_dirs:
                return
                
            debug_dir = self.client_debug_dirs[client_id]
            summary_path = os.path.join(debug_dir, "session_summary.txt")
            
            # Get session statistics
            total_frames = self.client_frame_counters.get(client_id, 0)
            buffer_frames = len(self.client_buffers.get(client_id, []))
            motion_history = list(self.client_motion_histories.get(client_id, []))
            
            # Calculate motion statistics
            avg_motion = sum(motion_history) / len(motion_history) if motion_history else 0
            max_motion = max(motion_history) if motion_history else 0
            min_motion = min(motion_history) if motion_history else 0
            
            with open(summary_path, "w") as f:
                f.write(f"WebSocket Session Summary\n")
                f.write(f"=========================\n")
                f.write(f"Client ID: {client_id}\n")
                f.write(f"Total Frames Received: {total_frames}\n")
                f.write(f"Buffer Size: {buffer_frames}/{config.SEQUENCE_LENGTH}\n")
                f.write(f"Motion Threshold: {self.motion_threshold}\n")
                f.write(f"Confidence Threshold: {self.confidence_threshold}\n\n")
                
                f.write(f"Motion Analysis:\n")
                f.write(f"----------------\n")
                f.write(f"Average Motion Score: {avg_motion:.6f}\n")
                f.write(f"Maximum Motion Score: {max_motion:.6f}\n")
                f.write(f"Minimum Motion Score: {min_motion:.6f}\n")
                f.write(f"Motion Above Threshold: {avg_motion > self.motion_threshold}\n\n")
                
                f.write(f"Configuration:\n")
                f.write(f"-------------\n")
                f.write(f"Input Size: {config.INPUT_SIZE}x{config.INPUT_SIZE}\n")
                f.write(f"Sequence Length: {config.SEQUENCE_LENGTH}\n")
                f.write(f"History Size: {config.HISTORY_SIZE}\n")
                
                # Add recent motion history
                if motion_history:
                    f.write(f"\nRecent Motion History (last 10):\n")
                    f.write(f"---------------------------------\n")
                    recent_motion = motion_history[-10:] if len(motion_history) > 10 else motion_history
                    for i, motion in enumerate(recent_motion, 1):
                        f.write(f"{i}. {motion:.6f} {'✓' if motion > self.motion_threshold else '✗'}\n")
            
            logger.info(f"Created session summary for {client_id}")
            
        except Exception as e:
            logger.error(f"Error creating session summary: {e}")
    
    def adjust_frame_rate(self, client_id: str, network_quality: float):
        """Dynamically adjust frame rate based on network quality"""
        if network_quality > 0.8:
            self.adaptive_fps[client_id] = 15  # High quality
        elif network_quality > 0.5:
            self.adaptive_fps[client_id] = 10  # Medium quality
        else:
            self.adaptive_fps[client_id] = 5   # Low quality
            
    def intelligent_frame_skip(self, client_id: str, frame_data: str) -> bool:
        """Skip frames intelligently based on motion and quality"""
        # Only process frames with significant changes
        # Skip redundant frames to save bandwidth
        return True  # Implement smart skipping logic
        
    async def handle_reconnection(self, client_id: str, websocket):
        """Handle client reconnections gracefully"""
        if client_id in self.client_sessions:
            # Restore previous session state
            logger.info(f"Restoring session for {client_id}")
            # Resume from where they left off

# Global detector instance
detector = OptimizedRealTimeDetector()

async def handle_client(websocket):
    """Handle WebSocket client connection"""
    client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}:{time.time()}"
    logger.info(f"Client connected: {client_id}")
    
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                
                if data.get("type") == "frame":
                    # Process frame
                    result = detector.process_frame(client_id, data.get("data", ""))
                    await websocket.send(json.dumps(result))
                
                elif data.get("type") == "ping":
                    # Respond to ping
                    await websocket.send(json.dumps({"type": "pong"}))
                
                elif data.get("type") == "adjust_threshold":
                    # Adjust motion threshold
                    factor = data.get("factor", 1.0)
                    detector.motion_threshold *= factor
                    await websocket.send(json.dumps({
                        "type": "threshold_updated",
                        "new_threshold": detector.motion_threshold
                    }))
                
            except json.JSONDecodeError:
                await websocket.send(json.dumps({
                    "status": "error", 
                    "message": "Invalid JSON"
                }))
            except Exception as e:
                logger.error(f"Message processing error: {e}")
                await websocket.send(json.dumps({
                    "status": "error", 
                    "message": str(e)
                }))
                
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Client disconnected: {client_id}")
    except Exception as e:
        logger.error(f"Client handling error: {e}")
    finally:
        detector.cleanup_client(client_id)

async def main():
    """Start the WebSocket server"""
    logger.info("Starting Real-Time Sign Language Detection WebSocket Server")
    logger.info(f"Server will listen on ws://0.0.0.0:8765")
    logger.info(f"Network access: ws://192.168.1.11:8765")
    logger.info(f"Local access: ws://localhost:8765")
    logger.info(f"Motion Threshold: {detector.motion_threshold}")
    logger.info(f"Confidence Threshold: {detector.confidence_threshold}")
    logger.info(f"Sequence Length: {config.SEQUENCE_LENGTH}")
    
    logger.info("Server started! Waiting for clients...")
    logger.info("Press Ctrl+C to stop the server")
      # Start the server and run forever    # Configure WebSocket server with appropriate limits for image streaming
    async with websockets.serve(
        handle_client, 
        "0.0.0.0", 
        8765,
        max_size=4 * 1024 * 1024,  # 4MB message limit (default is 1MB)
        ping_interval=30,          # Ping every 30 seconds
        ping_timeout=30,           # Wait 30 seconds for pong
        close_timeout=10           # Close connection timeout
    ):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise
