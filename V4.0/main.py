#!/usr/bin/env python3
import signal
import sys
import time
import os
import argparse
import yaml
import torch
from producer import FrameProducer
import logging
import threading
from datetime import datetime
import copy
import json

logger = logging.getLogger(__name__)

def setup_logging(config):
    """Configure logging to write to both file and console"""
    paths = config.get('paths', {})
    log_dir = paths.get('log_dir', './logs')
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"detection_{timestamp}.log")
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = []
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')
    
    root_logger.addHandler(console_handler)
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    return log_file

def select_consumer(config):
    """
    Select the appropriate consumer based on the final configuration.
    Returns the Consumer class and mode information.
    """
    consumer_mode = config.get('consumer', {}).get('mode', 'cpu')
    
    if consumer_mode == 'gpu' and torch.cuda.is_available():
        try:
            from consumer_gpu import GPUOptimizedConsumer
            logger.info("Using GPU-optimized consumer")
            return GPUOptimizedConsumer, 'gpu'
        except ImportError as e:
            logger.warning(f"Failed to import GPU consumer: {e}, falling back to CPU consumer")
            from consumer_cpu import CPUConsumer
            logger.info("Using standard CPU consumer")
            return CPUConsumer, 'cpu'
    else:
        from consumer_cpu import CPUConsumer
        logger.info("Using standard CPU consumer")
        return CPUConsumer, 'cpu'

class HealthAPI:
    """Optional Flask-based health monitoring API"""
    def __init__(self, app_instance, config, start_time):
        self.app_instance = app_instance
        self.config = config
        self.start_time = start_time
        self.flask_app = None
        self.api_thread = None
        
        health_config = config.get('health_api', {})
        self.enabled = health_config.get('enabled', False)
        self.host = health_config.get('host', '0.0.0.0')
        self.port = health_config.get('port', 8080)
        
        if self.enabled:
            try:
                from flask import Flask, jsonify
                self.Flask = Flask
                self.jsonify = jsonify
                self._setup_api()
            except ImportError:
                logger.error("Flask not installed. Install with: pip install flask")
                self.enabled = False
    
    def _setup_api(self):
        """Setup Flask routes"""
        self.flask_app = self.Flask(__name__)
        
        @self.flask_app.route('/status')
        def status():
            try:
                uptime_seconds = time.time() - self.start_time
                producer_camera_status = self.app_instance.producer.get_camera_status()
                
                camera_details = {}
                for camera_id, config in self.config['cameras'].items():
                    camera_details[camera_id] = {
                        'name': config['name'],
                        'rtsp_url': config['rtsp_url'],
                        'status': producer_camera_status.get(camera_id, {'connected': False, 'thread_alive': False})
                    }

                status_data = {
                    'system': 'running' if self.app_instance.running else 'stopped',
                    'uptime_seconds': uptime_seconds,
                    'queue_size': self.app_instance.producer.queue_size(),
                    'cameras': camera_details,
                    'consumer_status': self.app_instance.consumer.get_status()
                }
                return self.jsonify(status_data), 200
            except Exception as e:
                return self.jsonify({'error': str(e)}), 500
        
        @self.flask_app.route('/health')
        def health():
            uptime_seconds = time.time() - self.start_time
            return self.jsonify({'status': 'healthy', 'uptime_seconds': uptime_seconds}), 200
        
        @self.flask_app.route('/cameras')
        def cameras():
            try:
                camera_info = {}
                camera_status = self.app_instance.producer.get_camera_status()
                for camera_id, config in self.config['cameras'].items():
                    camera_info[camera_id] = {
                        'name': config['name'],
                        'rtsp_url': config['rtsp_url'],
                        'rois': self.config.get('regions_of_interest', {}).get(camera_id, {}),
                        'status': camera_status.get(camera_id, {'connected': False, 'thread_alive': False})
                    }
                return self.jsonify(camera_info), 200
            except Exception as e:
                return self.jsonify({'error': str(e)}), 500
    
    def start(self):
        """Start the Flask API in a separate thread"""
        if not self.enabled or not self.flask_app:
            return
        
        def run_api():
            logger.info(f"Starting health API on http://{self.host}:{self.port}")
            self.flask_app.run(host=self.host, port=self.port, debug=False, use_reloader=False)
        
        self.api_thread = threading.Thread(target=run_api, daemon=True)
        self.api_thread.start()
        logger.info(f"Health API endpoints:")
        logger.info(f"  - Status: http://{self.host}:{self.port}/status")
        logger.info(f"  - Health: http://{self.host}:{self.port}/health")
        logger.info(f"  - Cameras: http://{self.host}:{self.port}/cameras")
    
    def stop(self):
        """Stop the Flask API"""
        if self.api_thread and self.api_thread.is_alive():
            logger.info("Health API stopped")

class MultiCameraDetector:
    def __init__(self, config, visualize_mode=None):
        self.start_time = time.time()
        self.config = config
        self.visualize_mode = visualize_mode
        
        paths = self.config.get('paths', {})
        for path_value in paths.values():
            os.makedirs(path_value, exist_ok=True)
        
        ConsumerClass, self.consumer_mode = select_consumer(self.config)
        
        self.producer = FrameProducer(self.config)
        self.consumer = ConsumerClass(self.config, visualize_mode=visualize_mode)
        self.consumer_thread = threading.Thread(target=self.consumer.start, args=(self.producer,))
        self.running = True
        
        self.health_api = HealthAPI(self, self.config, self.start_time)

    def start(self):
        """Start the detection system"""
        viz_info = f" with {self.visualize_mode} mode" if self.visualize_mode else ""
        logger.info(f"Starting Multi-Camera Smoke Detection System{viz_info}")
        logger.info(f"Consumer mode: {self.consumer_mode.upper()}")
        
        if hasattr(self.consumer, 'enable_parallel'):
            if self.consumer.enable_parallel:
                logger.info(f"Parallel processing: ENABLED ({self.consumer.executor._max_workers} workers)")
            else:
                logger.info("Parallel processing: DISABLED")
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.health_api.start()
        
        logger.info("Starting Producer.")
        self.producer.start()

        logger.info("Starting Consumer thread.")
        self.consumer_thread.start()

        logger.info("Pipeline running. Press Ctrl+C to stop.")
        
        last_status_time = time.time()
        status_interval = 60  # Print status every 60 seconds
        
        while self.running:
            time.sleep(0.5)
            
            if not self.consumer_thread.is_alive():
                logger.error("Consumer thread has died. Stopping application.")
                self.running = False
                break
            
            if time.time() - last_status_time >= status_interval:
                self._print_status()
                last_status_time = time.time()

        logger.info("Pipeline stopping...")
        self.health_api.stop()
        self.consumer.stop()
        self.consumer_thread.join()
        self.producer.stop()
        logger.info("Pipeline stopped cleanly.")
    
    def _signal_handler(self, signum, frame):
        if self.running:
            logger.info(f"Signal {signum} received. Stopping...")
            self.running = False
    
    def _print_status(self):
        """Print current system status"""
        queue_size = self.producer.queue_size()
        camera_status = self.producer.get_camera_status()
        
        logger.info("="*80)
        logger.info("SYSTEM STATUS")
        logger.info(f"Uptime: {datetime.now() - datetime.fromtimestamp(self.start_time)}")
        logger.info(f"Consumer mode: {self.consumer_mode.upper()}")
        logger.info(f"Frame queue size: {queue_size}")
        
        logger.info("Camera Status:")
        for camera_id, status in camera_status.items():
            cam_name = self.config['cameras'][camera_id]['name']
            status_str = "Connected" if status['connected'] else "Disconnected"
            thread_str = "Running" if status['thread_alive'] else "Stopped"
            logger.info(f"  {cam_name} ({camera_id}): {status_str} | Thread: {thread_str}")
        
        logger.info("="*80)

def main():
    parser = argparse.ArgumentParser(
        description='Multi-Camera Smoke Detection and Notification System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --profile cpu               # Run with CPU-optimized profile
  python main.py --profile gpu               # Run with GPU-optimized profile
  python main.py --profile gpu --visualize   # Run with GPU profile and display results
  python main.py --profile cpu --save        # Run with CPU profile and save videos

Profiles:
  - The --profile argument selects a configuration profile from config.yaml.
  - 'cpu': Optimized for parallel CPU processing.
  - 'gpu': Optimized for batched GPU inference.
  - If --profile is not provided, the script will auto-detect the best
    mode, preferring GPU if a CUDA device is available.

Configuration:
  - All settings are in the 'config.yaml' file.
        """
    )
    
    parser.add_argument(
        '--profile',
        default='auto',
        help="Specifies the execution profile ('cpu', 'gpu', or 'auto'). Default: 'auto'."
    )
    
    viz_group = parser.add_mutually_exclusive_group()
    viz_group.add_argument('--visualize', action='store_true', help='Display detection results in real-time.')
    viz_group.add_argument('--save', action='store_true', help='Save detection visualization as video files.')
    
    args = parser.parse_args()
    
    config_path = 'config.yaml'
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {config_path}. Exiting.")
        sys.exit(1)

    profile_name = args.profile
    if profile_name == 'auto':
        profile_name = 'gpu' if torch.cuda.is_available() else 'cpu'
    
    final_config = copy.deepcopy(config)

    if profile_name in final_config.get('profiles', {}):
        profile_settings = final_config['profiles'][profile_name]
        # Deep merge
        for key, value in profile_settings.items():
            if isinstance(value, dict) and key in final_config:
                final_config[key].update(value)
            else:
                final_config[key] = value
        logger.info(f"Successfully loaded and merged '{profile_name}' profile.")
    else:
        logger.error(f"Profile '{profile_name}' not found in {config_path}. Exiting.")
        sys.exit(1)

    if 'profiles' in final_config:
        del final_config['profiles']
    
    log_file = setup_logging(final_config)
    logger.info(f"Logging to file: {log_file}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("CUDA not available, will use CPU.")
    
    visualize_mode = 'display' if args.visualize else 'save' if args.save else None
    
    try:
        app = MultiCameraDetector(config=final_config, visualize_mode=visualize_mode)
        app.start()
    except Exception as e:
        logger.error(f"Fatal error in application: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    # Change CWD to the script's directory to ensure relative paths work
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
