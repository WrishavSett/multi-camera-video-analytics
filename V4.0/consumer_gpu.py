import time
import yaml
import torch
from ultralytics import YOLO
import numpy as np
import cv2
import logging
import os
from datetime import datetime
import threading
from collections import defaultdict

from NotificationHandler import NotificationHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPUOptimizedConsumer:
    """
    GPU-optimized consumer for smoke detection and notification.
    Processes frames in batches for maximum GPU utilization.
    """

    def __init__(self, config, visualize_mode=None):
        self.config = config

        # Path configuration
        self.paths = self.config.get('paths', {})
        self.model_dir = self.paths.get('model_dir', './models')
        self.output_video_dir = self.paths.get('output_video_dir', './output_videos')
        
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Load YOLO model
        model_path = os.path.join(self.model_dir, self.config['model_name'])
        self.model = YOLO(model_path)
        self.device = self.config['device']
        self.batch_size = self.config['batch_size']
        self.batch_timeout = self.config['batch_timeout']
        self.confidence_threshold = self.config['confidence_threshold']
        self.model_input_size = tuple(self.config['model_input_size'])
        
        # Verify GPU availability
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"
        
        self.classes_to_track = set(self.config.get('classes_to_track', []))
        self.class_id_to_name = self.model.names
        
        # Initialize Notification Handler
        self.notification_handler = NotificationHandler(self.config)
        
        # Visualization settings
        self.visualize_mode = visualize_mode
        self.viz_config = self.config.get('visualization', {})
        self.video_config = self.config.get('video_save', {})
        self.video_writers = {}
        
        if self.visualize_mode == 'save':
            os.makedirs(self.output_video_dir, exist_ok=True)
            logger.info(f"Video output directory: {self.output_video_dir}")
        
        self.pending_frames = []
        self.last_batch_time = time.time()
        self.running = threading.Event()
        self.running.clear()
        
        logger.info(f"Model loaded on {self.device}")
        logger.info(f"Model input size: {self.model_input_size}")
        logger.info(f"GPU-optimized batching: ENABLED (batch_size={self.batch_size})")
        logger.info(f"Tracking classes: {self.classes_to_track}")
        logger.info(f"Visualization mode: {self.visualize_mode if self.visualize_mode else 'disabled'}")

    def start(self, producer):
        """Start consuming frames from producer"""
        self.running.set()
        self.producer = producer
        
        while self.running.is_set():
            self._collect_frames()
            
            if self._should_process_batch():
                self._process_batch_gpu_optimized()
            
            if self.visualize_mode == 'display':
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            time.sleep(0.001)
    
    def stop(self):
        """Stop the consumer"""
        self.running.clear()
        
        if self.pending_frames:
            self._process_batch_gpu_optimized()
        
        if self.visualize_mode == 'save':
            for writer in self.video_writers.values():
                writer.release()
        
        if self.visualize_mode == 'display':
            cv2.destroyAllWindows()
    
    def _collect_frames(self):
        frame_data = self.producer.get_frame()
        if frame_data:
            self.pending_frames.append(frame_data)
    
    def _should_process_batch(self):
        if not self.pending_frames:
            return False
        batch_full = len(self.pending_frames) >= self.batch_size
        timeout_reached = (time.time() - self.last_batch_time) >= self.batch_timeout
        return batch_full or timeout_reached
    
    def _process_batch_gpu_optimized(self):
        if not self.pending_frames:
            return
        
        batch_start_time = time.time()
        
        batch_frames = [f['frame'] for f in self.pending_frames]
        frame_metadata = self.pending_frames
        
        try:
            results = self.model.track(
                batch_frames,
                conf=self.confidence_threshold,
                device=self.device,
                persist=True,
                verbose=False,
                imgsz=self.model_input_size[0]
            )
            
            for result, metadata in zip(results, frame_metadata):
                camera_id = metadata['camera_id']
                tracks = self._extract_tracks(result)
                
                if tracks:
                    self.notification_handler.check_and_notify(tracks, camera_id)
                
                if self.visualize_mode:
                    self._handle_visualization(metadata, tracks, camera_id)
            
        except Exception as e:
            logger.error(f"Error in GPU batch processing: {e}", exc_info=True)
        
        finally:
            self.pending_frames.clear()
            self.last_batch_time = time.time()
    
    def _extract_tracks(self, result):
        if result.boxes is None or len(result.boxes) == 0:
            return []
        
        if not hasattr(result.boxes, 'id') or result.boxes.id is None:
            return []

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        track_ids = result.boxes.id.cpu().numpy().astype(int)
        
        tracks = []
        for box, conf, cls_id, track_id in zip(boxes, confidences, class_ids, track_ids):
            class_name = self.class_id_to_name[cls_id]
            
            if self.classes_to_track and class_name not in self.classes_to_track:
                continue
            
            tracks.append({
                'bbox': [int(b) for b in box],
                'track_id': track_id,
                'class': class_name,
                'confidence': conf
            })
        return tracks
    
    def _handle_visualization(self, metadata, tracks, camera_id):
        original_frame = metadata['original_frame']
        display_frame = cv2.resize(original_frame, self.model_input_size)
        
        annotated_frame = self._draw_roi_boxes(display_frame, camera_id)
        annotated_frame = self._draw_tracks(annotated_frame, tracks, metadata['camera_name'])
        
        if self.visualize_mode == 'display':
            cv2.imshow(metadata['camera_name'], annotated_frame)
        elif self.visualize_mode == 'save':
            self._save_frame_to_video(annotated_frame, camera_id, metadata['camera_name'])
    
    def _draw_roi_boxes(self, frame, camera_id):
        if not self.viz_config.get('show_roi_boxes', True):
            return frame
        
        rois = self.config.get('regions_of_interest', {}).get(camera_id, {})
        color = tuple(self.viz_config.get('roi_color', [0, 255, 255]))
        thickness = self.viz_config.get('roi_thickness', 1)
        
        for chimney_id, roi in rois.items():
            x1, y1, x2, y2 = roi
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(frame, chimney_id, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
        
        return frame
    
    def _draw_tracks(self, frame, tracks, camera_name):
        box_color = tuple(self.viz_config.get('box_color', [0, 0, 255]))
        text_color = tuple(self.viz_config.get('text_color', [0, 0, 255]))
        box_thickness = self.viz_config.get('box_thickness', 2)
        text_thickness = self.viz_config.get('text_thickness', 2)
        font_scale = self.viz_config.get('font_scale', 0.6)
        
        for track in tracks:
            x1, y1, x2, y2 = track['bbox']
            label = f"ID:{track['track_id']} {track['class']} {track['confidence']:.2f}"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, box_thickness)
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
            cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), box_color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), text_thickness)
        
        info_y = 30
        cv2.putText(frame, camera_name, (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return frame
    
    def _save_frame_to_video(self, frame, camera_id, camera_name):
        if camera_id not in self.video_writers:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{camera_id}_{camera_name.replace(' ', '_')}_{timestamp}.mp4"
            filepath = os.path.join(self.output_video_dir, filename)
            
            height, width = frame.shape[:2]
            fps = self.video_config.get('fps', 8)
            codec = self.video_config.get('codec', 'mp4v')
            fourcc = cv2.VideoWriter_fourcc(*codec)
            
            writer = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
            if not writer.isOpened():
                logger.error(f"Failed to create video writer for {camera_name}: {filepath}")
                return
            
            self.video_writers[camera_id] = writer
            logger.info(f"Created video writer for {camera_name}: {filepath}")
        
        self.video_writers[camera_id].write(frame)

    def get_status(self):
        """Provides a basic status for health monitoring."""
        # This can be expanded to report on notification rates, etc.
        status = {
            'timestamp': datetime.now().isoformat(),
            'last_known_batch_size': len(self.pending_frames)
        }
        return status
