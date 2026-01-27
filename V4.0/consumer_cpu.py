import time
import torch
from ultralytics import YOLO
import cv2
import logging
import os
from datetime import datetime
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from NotificationHandler import NotificationHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CPUConsumer:
    """
    CPU-optimized consumer for smoke detection and notification.
    Uses a thread pool to process multiple camera streams in parallel.
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
        self.confidence_threshold = self.config['confidence_threshold']
        self.model_input_size = tuple(self.config['model_input_size'])
        
        self.classes_to_track = set(self.config.get('classes_to_track', []))
        self.class_id_to_name = self.model.names

        # Initialize Notification Handler
        self.notification_handler = NotificationHandler(self.config)
        
        # Parallel processing settings
        consumer_config = self.config.get('consumer', {})
        self.enable_parallel = consumer_config.get('enable_parallel', True)
        max_workers = consumer_config.get('max_workers', 4)
        self.executor = None
        if self.enable_parallel:
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
            logger.info(f"Parallel processing enabled with {max_workers} workers")
        
        # Visualization settings
        self.visualize_mode = visualize_mode
        self.viz_config = self.config.get('visualization', {})
        self.video_config = self.config.get('video_save', {})
        self.video_writers = {}
        
        if self.visualize_mode == 'save':
            os.makedirs(self.output_video_dir, exist_ok=True)
            logger.info(f"Video output directory: {self.output_video_dir}")

        self.pending_frames = []
        self.running = threading.Event()
        self.running.clear()
        
        logger.info(f"Model loaded on {self.device}")
        logger.info(f"Model input size: {self.model_input_size}")
        logger.info(f"Tracking classes: {self.classes_to_track}")
        logger.info(f"Visualization mode: {self.visualize_mode if self.visualize_mode else 'disabled'}")
    
    def start(self, producer):
        self.running.set()
        self.producer = producer
        
        while self.running.is_set():
            frame_data = self.producer.get_frame()
            if frame_data:
                self.pending_frames.append(frame_data)
            
            # Process frames as they come, don't wait for a full batch
            if self.pending_frames:
                self._process_frames()

            if self.visualize_mode == 'display':
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            time.sleep(0.001)

    def stop(self):
        self.running.clear()
        
        if self.executor:
            self.executor.shutdown(wait=True)
        
        if self.visualize_mode == 'save':
            for writer in self.video_writers.values():
                writer.release()
        
        if self.visualize_mode == 'display':
            cv2.destroyAllWindows()

    def _process_frames(self):
        if not self.pending_frames:
            return

        camera_frames = defaultdict(list)
        for frame_data in self.pending_frames:
            camera_frames[frame_data['camera_id']].append(frame_data)
        
        if self.enable_parallel and self.executor:
            # Parallel processing
            futures = {self.executor.submit(self._process_camera_frames, cid, frames) for cid, frames in camera_frames.items()}
            
            for future in as_completed(futures):
                try:
                    camera_id, results = future.result()
                    for result in results:
                        self._post_process(result)
                except Exception as e:
                    logger.error(f"Error in parallel processing: {e}", exc_info=True)
        else:
            # Sequential processing
            for cid, frames in camera_frames.items():
                camera_id, results = self._process_camera_frames(cid, frames)
                for result in results:
                    self._post_process(result)
        
        self.pending_frames.clear()

    def _process_camera_frames(self, camera_id, frames_data):
        """Worker function to process all frames for a single camera."""
        results_list = []
        for frame_data in frames_data:
            im0 = frame_data['frame']
            
            # Perform inference and tracking
            result = self.model.track(im0, persist=True, verbose=False, device=self.device)[0]
            tracks = self._extract_tracks(result)
            
            results_list.append({
                'camera_id': camera_id,
                'metadata': frame_data,
                'tracks': tracks
            })
        return camera_id, results_list

    def _post_process(self, result):
        """Handle notification and visualization for a processed frame."""
        camera_id = result['camera_id']
        tracks = result['tracks']
        metadata = result['metadata']

        if tracks:
            self.notification_handler.check_and_notify(tracks, camera_id)
        
        if self.visualize_mode:
            self._handle_visualization(metadata, tracks, camera_id)

    def _extract_tracks(self, result):
        # Same as GPU consumer
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
        # Same as GPU consumer
        original_frame = metadata['original_frame']
        display_frame = cv2.resize(original_frame, self.model_input_size)
        
        annotated_frame = self._draw_roi_boxes(display_frame, camera_id)
        annotated_frame = self._draw_tracks(annotated_frame, tracks, metadata['camera_name'])
        
        if self.visualize_mode == 'display':
            cv2.imshow(metadata['camera_name'], annotated_frame)
        elif self.visualize_mode == 'save':
            self._save_frame_to_video(annotated_frame, camera_id, metadata['camera_name'])

    def _draw_roi_boxes(self, frame, camera_id):
        # Same as GPU consumer
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
        # Same as GPU consumer
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
        # Same as GPU consumer
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
        status = {
            'timestamp': datetime.now().isoformat(),
            'pending_frames_in_queue': len(self.pending_frames)
        }
        return status
