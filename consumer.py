import time
import yaml
import torch
from ultralytics import YOLO
import numpy as np
import cv2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BatchConsumer:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Load YOLOv8 model
        self.model = YOLO(self.config['model_path'])
        self.device = self.config['device']
        self.batch_size = self.config['batch_size']
        self.batch_timeout = self.config['batch_timeout']
        self.confidence_threshold = self.config['confidence_threshold']
        self.model_input_size = tuple(self.config['model_input_size'])
        
        # Create class name to ID mapping for efficient filtering
        self.class_name_to_id = {name: idx for idx, name in self.model.names.items()}
        self.class_id_to_name = self.model.names
        
        self.pending_frames = []
        self.last_batch_time = time.time()
        self.running = False
        
        logger.info(f"Model loaded on {self.device}")
        logger.info(f"Model input size: {self.model_input_size}")
        logger.info(f"Available classes: {list(self.class_name_to_id.keys())}")
    
    def start(self, producer):
        """Start consuming frames from producer"""
        self.running = True
        self.producer = producer
        
        while self.running:
            # Collect frames for batch
            self._collect_frames()
            
            # Process batch if ready
            if self._should_process_batch():
                self._process_batch()
            
            time.sleep(0.001)  # Small sleep to prevent busy waiting
    
    def stop(self):
        """Stop the consumer"""
        self.running = False
        
        # Process any remaining frames
        if self.pending_frames:
            self._process_batch()
    
    def _collect_frames(self):
        """Collect frames from producer"""
        frame_data = self.producer.get_frame()
        if frame_data:
            self.pending_frames.append(frame_data)
    
    def _should_process_batch(self):
        """Determine if batch should be processed"""
        if not self.pending_frames:
            return False
        
        # Process if batch is full or timeout reached
        batch_full = len(self.pending_frames) >= self.batch_size
        timeout_reached = (time.time() - self.last_batch_time) >= self.batch_timeout
        
        return batch_full or timeout_reached
    
    def _get_class_ids_for_camera(self, classes_to_detect):
        """Convert class names to class IDs for filtering"""
        if not classes_to_detect:  # Empty list means detect all classes
            return None
        
        class_ids = []
        invalid_classes = []
        
        for class_name in classes_to_detect:
            if class_name in self.class_name_to_id:
                class_ids.append(self.class_name_to_id[class_name])
            else:
                invalid_classes.append(class_name)
        
        if invalid_classes:
            logger.warning(f"Invalid class names found: {invalid_classes}. "
                          f"Available classes: {list(self.class_name_to_id.keys())}")
        
        return class_ids if class_ids else None
    
    def _filter_detections_by_class(self, detections, allowed_class_ids):
        """Filter detections by allowed class IDs"""
        if allowed_class_ids is None:  # None means all classes allowed
            return detections
        
        filtered_detections = []
        for detection in detections:
            if detection['class_id'] in allowed_class_ids:
                filtered_detections.append(detection)
        
        return filtered_detections
    
    def _filter_detections_by_roi(self, detections, roi_config, frame_shape):
        """Filter detections to only include those within ROI"""
        if not roi_config['enabled'] or not roi_config['coordinates']:
            return detections  # Return all detections if ROI is disabled
        
        filtered_detections = []
        
        # Create ROI mask for checking detection centers
        mask = np.zeros((frame_shape[0], frame_shape[1]), dtype=np.uint8)
        roi_points = np.array(roi_config['coordinates'], dtype=np.int32)
        cv2.fillPoly(mask, [roi_points], 255)
        
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            # Calculate center point of detection
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # Check if center point is within ROI
            if (0 <= center_y < mask.shape[0] and 
                0 <= center_x < mask.shape[1] and 
                mask[center_y, center_x] > 0):
                filtered_detections.append(detection)
            else:
                logger.debug(f"Detection filtered out - center ({center_x}, {center_y}) outside ROI")
        
        return filtered_detections
    
    def _process_batch(self):
        """Process batch of frames through YOLOv8"""
        if not self.pending_frames:
            return
        
        batch_start_time = time.time()
        
        # Prepare batch
        batch_frames = []
        frame_metadata = []
        
        for frame_data in self.pending_frames:
            batch_frames.append(frame_data['frame'])
            frame_metadata.append({
                'camera_id': frame_data['camera_id'],
                'camera_name': frame_data['camera_name'],
                'roi_config': frame_data['roi_config'],
                'classes_to_detect': frame_data['classes_to_detect'],
                'timestamp': frame_data['timestamp'],
                'frame_shape': frame_data['frame'].shape
            })
        
        try:
            # Run inference on batch (frames already have ROI applied)
            results = self.model(
                batch_frames,
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False
            )
            
            # Process results
            self._handle_results(results, frame_metadata)
            
            processing_time = time.time() - batch_start_time
            logger.info(f"Processed batch of {len(batch_frames)} frames in {processing_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
        
        finally:
            # Clear batch
            self.pending_frames.clear()
            self.last_batch_time = time.time()
    
    def _handle_results(self, results, frame_metadata):
        """Handle inference results with class and ROI filtering"""
        for i, (result, metadata) in enumerate(zip(results, frame_metadata)):
            detections = []
            
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                for box, conf, cls_id in zip(boxes, confidences, class_ids):
                    x1, y1, x2, y2 = box
                    class_name = self.class_id_to_name[cls_id]
                    
                    detection = {
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(conf),
                        'class': class_name,
                        'class_id': int(cls_id)
                    }
                    
                    detections.append(detection)
            
            # Filter by allowed classes for this camera
            allowed_class_ids = self._get_class_ids_for_camera(metadata['classes_to_detect'])
            class_filtered_detections = self._filter_detections_by_class(detections, allowed_class_ids)
            
            # Apply ROI filtering
            final_detections = self._filter_detections_by_roi(
                class_filtered_detections, 
                metadata['roi_config'], 
                metadata['frame_shape']
            )
            
            # Log results with filtering information
            if detections:
                class_filter_info = f"classes: {metadata['classes_to_detect']}" if metadata['classes_to_detect'] else "all classes"
                roi_info = "with ROI" if metadata['roi_config']['enabled'] else "full frame"
                
                logger.info(f"{metadata['camera_name']} ({roi_info}, {class_filter_info}): "
                          f"Raw: {len(detections)}, Class filtered: {len(class_filtered_detections)}, "
                          f"Final: {len(final_detections)}")
                
                for det in final_detections:
                    logger.info(f"  ✓ {det['class']}: {det['confidence']:.2f} at {det['bbox']}")
            
            # Send filtered results to downstream system
            self._send_results(metadata, final_detections, {
                'total_detections': len(detections),
                'class_filtered': len(class_filtered_detections),
                'final_detections': len(final_detections)
            })
    
    def _send_results(self, metadata, detections, filtering_stats):
        """Send results to downstream system"""
        result_summary = {
            'camera_id': metadata['camera_id'],
            'camera_name': metadata['camera_name'],
            'timestamp': metadata['timestamp'],
            'roi_enabled': metadata['roi_config']['enabled'],
            'roi_coordinates': metadata['roi_config']['coordinates'] if metadata['roi_config']['enabled'] else None,
            'classes_to_detect': metadata['classes_to_detect'],
            'filtering_stats': filtering_stats,
            'detection_count': len(detections),
            'detections': detections
        }
        
        # Here you can implement your result forwarding logic
        # For example: save to database, send to API, write to file, etc.

        # Optional: Save detection results to file for debugging
        if detections:
            self._save_detection_results(result_summary)
    
    def _save_detection_results(self, result_summary):
        """Optional: Save detection results for debugging/analysis"""
        import json
        from datetime import datetime
        
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"detections_{result_summary['camera_id']}_{timestamp_str}.json"
        
        try:
            with open(f"./logs/{filename}", 'w') as f:
                json.dump(result_summary, f, indent=2)
        except:
            pass  # Don't fail if logging directory doesn't exist