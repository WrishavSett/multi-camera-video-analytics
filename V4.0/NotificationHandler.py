import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
import time
from collections import defaultdict

logger = logging.getLogger(__name__)

class NotificationHandler:
    def __init__(self, config):
        self.email_config = config.get('email_notifications', {})
        self.chimney_rois = config.get('regions_of_interest', {})
        self.notification_cooldown = config.get('notification_cooldown', 300)  # 5 minutes default
        
        # State to track notifications {chimney_id: {track_id: last_notification_time}}
        self.notification_log = defaultdict(lambda: defaultdict(float))
        
        if not self.email_config.get('enabled', False):
            logger.warning("Email notifications are disabled in the configuration.")

    def check_and_notify(self, tracks, camera_id):
        """
        Check for new smoke emissions in chimney ROIs and send notifications.

        Args:
            tracks (list): A list of dictionaries, where each dict represents a tracked object.
                           Expected keys: 'bbox', 'class', 'track_id'.
            camera_id (str): The ID of the camera that produced the tracks.
        """
        if not self.email_config.get('enabled', False):
            return

        current_time = time.time()
        camera_chimneys = self.chimney_rois.get(camera_id, {})

        if not camera_chimneys:
            logger.debug(f"No chimney ROIs defined for camera {camera_id}.")
            return

        for track in tracks:
            if track['class'] != 'smoke':  # Assuming the model class for smoke is 'smoke'
                continue

            track_id = track['track_id']
            x1, y1, x2, y2 = track['bbox']
            track_centroid = ((x1 + x2) / 2, (y1 + y2) / 2)

            for chimney_id, roi_coords in camera_chimneys.items():
                roi_x1, roi_y1, roi_x2, roi_y2 = roi_coords
                
                # Check if the track's centroid is within the chimney's ROI
                if roi_x1 < track_centroid[0] < roi_x2 and roi_y1 < track_centroid[1] < roi_y2:
                    
                    last_notification_time = self.notification_log[chimney_id].get(track_id, 0)
                    
                    # Check if a new notification is warranted
                    if current_time - last_notification_time > self.notification_cooldown:
                        logger.info(f"New emission event detected for Chimney {chimney_id} (Track ID: {track_id}). Sending notification.")
                        
                        # Update notification log BEFORE sending to prevent race conditions
                        self.notification_log[chimney_id][track_id] = current_time
                        
                        self._send_email(chimney_id, camera_id, track)
                        break # A track can only be in one chimney at a time

    def _send_email(self, chimney_id, camera_id, track_details):
        """
        Composes and sends an email notification.
        """
        smtp_server = self.email_config['smtp_server']
        smtp_port = self.email_config['smtp_port']
        sender_email = self.email_config['sender_email']
        password = self.email_config['password']
        recipient_emails = self.email_config['recipients']

        subject = f"Smoke Emission Alert: Chimney {chimney_id}"
        body = f"""
        Alert,

        A new smoke emission event has been detected.

        Details:
        - Camera: {camera_id}
        - Chimney Number: {chimney_id}
        - Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}
        - Track ID: {track_details['track_id']}
        - Confidence: {track_details.get('confidence', 'N/A'):.2f}

        This is an automated notification.
        """

        try:
            with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
                server.login(sender_email, password)
                
                for recipient in recipient_emails:
                    msg = MIMEMultipart()
                    msg['From'] = sender_email
                    msg['To'] = recipient
                    msg['Subject'] = subject
                    msg.attach(MIMEText(body, 'plain'))
                    
                    server.send_message(msg)
                    logger.info(f"Notification email sent successfully to {recipient} for Chimney {chimney_id}.")

        except Exception as e:
            logger.error(f"Failed to send email for Chimney {chimney_id}: {e}", exc_info=True)
