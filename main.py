#!/usr/bin/env python3
import signal
import sys
import time
import os
import argparse
from producer import FrameProducer
from consumer import BatchConsumer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiCameraAnalytics:
    def __init__(self, visualize_mode=None):
        self.producer = FrameProducer()
        self.consumer = BatchConsumer(visualize_mode=visualize_mode)
        self.running = False
        self.visualize_mode = visualize_mode
        
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # Ensure models directory exists
        os.makedirs("models", exist_ok=True)
    
    def start(self):
        """Start the analytics system"""
        viz_info = f" with {self.visualize_mode} mode" if self.visualize_mode else ""
        logger.info(f"Starting Multi-Camera Analytics System with ROI and Class Filtering{viz_info}")
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        try:
            # Start producer
            self.producer.start()
            time.sleep(2)  # Give cameras time to initialize
            
            # Start consumer
            self.running = True
            logger.info("System started successfully - Processing frames with ROI and class constraints")
            self.consumer.start(self.producer)
            
        except Exception as e:
            logger.error(f"Error starting system: {e}")
            self.stop()
    
    def stop(self):
        """Stop the analytics system"""
        logger.info("Shutting down system...")
        self.running = False
        
        if hasattr(self, 'consumer'):
            self.consumer.stop()
        
        if hasattr(self, 'producer'):
            self.producer.stop()
        
        logger.info("System shutdown complete")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}")
        self.stop()
        sys.exit(0)
    
    def status(self):
        """Print system status"""
        queue_size = self.producer.queue_size()
        logger.info(f"Frame queue size: {queue_size}")

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description='Multi-Camera Video Analytics System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run without visualization
  python main.py --visualize        # Display detections in OpenCV windows
  python main.py --save             # Save annotated videos to disk
        """
    )
    
    viz_group = parser.add_mutually_exclusive_group()
    viz_group.add_argument(
        '--visualize',
        action='store_true',
        help='Display detection results in real-time OpenCV windows'
    )
    viz_group.add_argument(
        '--save',
        action='store_true',
        help='Save detection results as video files'
    )
    
    args = parser.parse_args()
    
    # Determine visualization mode
    visualize_mode = None
    if args.visualize:
        visualize_mode = 'display'
        logger.info("Visualization mode: Real-time display")
    elif args.save:
        visualize_mode = 'save'
        logger.info("Visualization mode: Save to video files")
    else:
        logger.info("Visualization mode: Disabled (logging only)")
    
    # Initialize and start the system
    app = MultiCameraAnalytics(visualize_mode=visualize_mode)
    
    try:
        app.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        app.stop()

if __name__ == "__main__":
    main()