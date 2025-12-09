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

    def send_notification(self, to_email: str, subject: str, body: str, bcc: list[str] = None):
        if self.mock_mode:
            self._log_email(to_email, subject, body, bcc)
        else:
            self._send_smtp_email(to_email, subject, body, bcc)

    def _log_email(self, to_email: str, subject: str, body: str, bcc: list[str] = None):
        """Mock send by logging"""
        bcc_str = f"BCC: {', '.join(bcc)}\n" if bcc else ""
        log_message = (
            f"\n{'='*50}\n"
            f"TO: {to_email}\n"
            f"{bcc_str}"
            f"SUBJECT: {subject}\n"
            f"TIME: {datetime.now()}\n"
            f"{'-'*50}\n"
            f"{body}\n"
            f"{'='*50}\n"
        )
        logger.info(f"MOCK EMAIL SENT: {log_message}")
        print(log_message) # Ensure it hits stdout

    def _send_smtp_email(self, to_email: str, subject: str, body: str, bcc: list[str] = None):
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart

        try:
            msg = MIMEMultipart()
            msg['From'] = self.smtp_user
            msg['To'] = to_email
            msg['Subject'] = subject
            # BCC headers are NOT added to the message object itself to keep them blind
            # msg['Bcc'] = ... (DO NOT DO THIS)

            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.smtp_user, self.smtp_password)
            text = msg.as_string()
            
            # Combine all recipients for the envelope
            recipients = [to_email]
            if bcc:
                recipients.extend(bcc)
                
            server.sendmail(self.smtp_user, recipients, text)
            server.quit()
            logger.info(f"Email sent successfully to {to_email} (BCC: {len(bcc) if bcc else 0})")
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            # Fallback to log if SMTP fails
            self._log_email(to_email, subject, f"[FALLBACK due to error: {e}]\n{body}")

# Global instance
email_service = EmailService()
