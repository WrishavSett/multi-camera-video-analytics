#!/usr/bin/env python3
import signal
import sys
import time
import os
from producer import FrameProducer
from consumer import BatchConsumer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiCameraAnalytics:
    def __init__(self):
        self.producer = FrameProducer()
        self.consumer = BatchConsumer()
        self.running = False
        
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # Ensure models directory exists
        os.makedirs("models", exist_ok=True)
    
    def start(self):
        """Start the analytics system"""
        logger.info("Starting Multi-Camera Analytics System with ROI and Class Filtering")
        
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

if __name__ == "__main__":
    app = MultiCameraAnalytics()
    
    try:
        app.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        app.stop()