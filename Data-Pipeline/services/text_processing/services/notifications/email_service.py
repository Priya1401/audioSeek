import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class EmailService:
    def __init__(self):
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = os.getenv("SMTP_USER")
        self.smtp_password = os.getenv("SMTP_PASSWORD")
        
        # If no credentials, default to mock mode
        self.mock_mode = not (self.smtp_user and self.smtp_password)
        if self.mock_mode:
            logger.info("EmailService initialized in MOCK mode. Emails will be logged to console.")
        else:
            logger.info("EmailService initialized in SMTP mode.")

    def send_notification(self, to_email: str, subject: str, body: str):
        if self.mock_mode:
            self._log_email(to_email, subject, body)
        else:
            self._send_smtp_email(to_email, subject, body)

    def _log_email(self, to_email: str, subject: str, body: str):
        """Mock send by logging"""
        log_message = (
            f"\n{'='*50}\n"
            f"TO: {to_email}\n"
            f"SUBJECT: {subject}\n"
            f"TIME: {datetime.now()}\n"
            f"{'-'*50}\n"
            f"{body}\n"
            f"{'='*50}\n"
        )
        logger.info(f"MOCK EMAIL SENT: {log_message}")
        print(log_message) # Ensure it hits stdout

    def _send_smtp_email(self, to_email: str, subject: str, body: str):
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart

        try:
            msg = MIMEMultipart()
            msg['From'] = self.smtp_user
            msg['To'] = to_email
            msg['Subject'] = subject

            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.smtp_user, self.smtp_password)
            text = msg.as_string()
            server.sendmail(self.smtp_user, to_email, text)
            server.quit()
            logger.info(f"Email sent successfully to {to_email}")
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            # Fallback to log if SMTP fails
            self._log_email(to_email, subject, f"[FALLBACK due to error: {e}]\n{body}")

# Global instance
email_service = EmailService()
