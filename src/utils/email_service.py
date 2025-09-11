"""Email notification service for enterprise features."""

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import List, Optional, Dict, Any
import os
from datetime import datetime
from jinja2 import Template
from src.utils.logger import logger


class EmailService:
    """Email notification service with template support."""
    
    def __init__(self, smtp_host: str = None, smtp_port: int = 587, 
                 smtp_user: str = None, smtp_password: str = None):
        """Initialize email service."""
        self.smtp_host = smtp_host or os.getenv('SMTP_HOST', 'localhost')
        self.smtp_port = smtp_port or int(os.getenv('SMTP_PORT', '587'))
        self.smtp_user = smtp_user or os.getenv('SMTP_USER')
        self.smtp_password = smtp_password or os.getenv('SMTP_PASSWORD')
        self.enabled = bool(self.smtp_user and self.smtp_password)
        
        if not self.enabled:
            logger.warning("Email service disabled - missing SMTP credentials")
    
    def send_email(self, to_emails: List[str], subject: str, 
                   html_content: str = None, text_content: str = None,
                   attachments: List[Dict[str, Any]] = None) -> bool:
        """Send email to recipients."""
        if not self.enabled:
            logger.warning("Email service disabled")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['From'] = self.smtp_user
            msg['To'] = ', '.join(to_emails)
            msg['Subject'] = subject
            
            # Add text content
            if text_content:
                text_part = MIMEText(text_content, 'plain')
                msg.attach(text_part)
            
            # Add HTML content
            if html_content:
                html_part = MIMEText(html_content, 'html')
                msg.attach(html_part)
            
            # Add attachments
            if attachments:
                for attachment in attachments:
                    self._add_attachment(msg, attachment)
            
            # Send email
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)
            
            logger.info(f"Email sent successfully to {len(to_emails)} recipients")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False
    
    def _add_attachment(self, msg: MIMEMultipart, attachment: Dict[str, Any]):
        """Add attachment to email message."""
        try:
            filename = attachment['filename']
            content = attachment['content']
            content_type = attachment.get('content_type', 'application/octet-stream')
            
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(content)
            encoders.encode_base64(part)
            part.add_header(
                'Content-Disposition',
                f'attachment; filename= {filename}'
            )
            msg.attach(part)
        except Exception as e:
            logger.error(f"Failed to add attachment: {e}")
    
    def send_welcome_email(self, client_email: str, client_name: str, 
                          company_name: str) -> bool:
        """Send welcome email to new client."""
        subject = f"Welcome to Video Insights Platform - {company_name}"
        
        html_template = """
        <html>
        <body>
            <h2>Welcome to Video Insights Platform!</h2>
            <p>Dear {{ client_name }},</p>
            <p>Welcome to the Video Insights Platform! Your account for {{ company_name }} has been successfully created.</p>
            
            <h3>What you can do:</h3>
            <ul>
                <li>Analyze YouTube videos with AI-powered insights</li>
                <li>Track usage analytics and performance metrics</li>
                <li>Generate detailed reports for your team</li>
                <li>Access enterprise-grade features and support</li>
            </ul>
            
            <p>Get started by visiting our platform and uploading your first video for analysis.</p>
            
            <p>If you have any questions, please don't hesitate to contact our support team.</p>
            
            <p>Best regards,<br>
            Video Insights Team</p>
        </body>
        </html>
        """
        
        text_template = """
        Welcome to Video Insights Platform!
        
        Dear {{ client_name }},
        
        Welcome to the Video Insights Platform! Your account for {{ company_name }} has been successfully created.
        
        What you can do:
        - Analyze YouTube videos with AI-powered insights
        - Track usage analytics and performance metrics
        - Generate detailed reports for your team
        - Access enterprise-grade features and support
        
        Get started by visiting our platform and uploading your first video for analysis.
        
        If you have any questions, please don't hesitate to contact our support team.
        
        Best regards,
        Video Insights Team
        """
        
        template = Template(html_template)
        html_content = template.render(
            client_name=client_name,
            company_name=company_name
        )
        
        template = Template(text_template)
        text_content = template.render(
            client_name=client_name,
            company_name=company_name
        )
        
        return self.send_email(
            to_emails=[client_email],
            subject=subject,
            html_content=html_content,
            text_content=text_content
        )
    
    def send_quota_warning_email(self, client_email: str, client_name: str,
                                usage_percentage: float, quota: int) -> bool:
        """Send quota warning email."""
        subject = f"API Quota Warning - {usage_percentage:.1f}% Used"
        
        html_template = """
        <html>
        <body>
            <h2>API Quota Warning</h2>
            <p>Dear {{ client_name }},</p>
            <p>Your API quota usage has reached {{ usage_percentage }}% ({{ current_usage }}/{{ quota }} requests).</p>
            
            <div style="background-color: #fff3cd; padding: 15px; border-radius: 5px; margin: 20px 0;">
                <strong>Current Usage:</strong> {{ current_usage }} / {{ quota }} requests<br>
                <strong>Usage Percentage:</strong> {{ usage_percentage }}%
            </div>
            
            <p>To avoid service interruption, please consider:</p>
            <ul>
                <li>Upgrading your subscription plan</li>
                <li>Optimizing your usage patterns</li>
                <li>Contacting support for assistance</li>
            </ul>
            
            <p>Best regards,<br>
            Video Insights Team</p>
        </body>
        </html>
        """
        
        current_usage = int(quota * usage_percentage / 100)
        
        template = Template(html_template)
        html_content = template.render(
            client_name=client_name,
            usage_percentage=usage_percentage,
            current_usage=current_usage,
            quota=quota
        )
        
        return self.send_email(
            to_emails=[client_email],
            subject=subject,
            html_content=html_content
        )
    
    def send_usage_report_email(self, client_email: str, client_name: str,
                               report_data: Dict[str, Any]) -> bool:
        """Send monthly usage report email."""
        subject = f"Monthly Usage Report - {datetime.now().strftime('%B %Y')}"
        
        html_template = """
        <html>
        <body>
            <h2>Monthly Usage Report</h2>
            <p>Dear {{ client_name }},</p>
            <p>Here's your monthly usage report for {{ month_year }}:</p>
            
            <table style="border-collapse: collapse; width: 100%; margin: 20px 0;">
                <tr style="background-color: #f2f2f2;">
                    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Metric</th>
                    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Value</th>
                </tr>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px;">Total Queries</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{{ total_queries }}</td>
                </tr>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px;">Unique Videos</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{{ unique_videos }}</td>
                </tr>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px;">Average Response Time</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{{ avg_response_time }}s</td>
                </tr>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px;">API Usage</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{{ api_usage }}/{{ api_quota }}</td>
                </tr>
            </table>
            
            <p>Thank you for using Video Insights Platform!</p>
            
            <p>Best regards,<br>
            Video Insights Team</p>
        </body>
        </html>
        """
        
        template = Template(html_template)
        html_content = template.render(
            client_name=client_name,
            month_year=datetime.now().strftime('%B %Y'),
            total_queries=report_data.get('total_queries', 0),
            unique_videos=report_data.get('unique_videos', 0),
            avg_response_time=report_data.get('avg_response_time', 0),
            api_usage=report_data.get('api_usage', 0),
            api_quota=report_data.get('api_quota', 0)
        )
        
        return self.send_email(
            to_emails=[client_email],
            subject=subject,
            html_content=html_content
        )
    
    def send_error_alert_email(self, admin_emails: List[str], error_message: str,
                              client_id: str = None) -> bool:
        """Send error alert email to administrators."""
        subject = f"System Error Alert - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        html_template = """
        <html>
        <body>
            <h2>System Error Alert</h2>
            <p>An error has occurred in the Video Insights Platform:</p>
            
            <div style="background-color: #f8d7da; padding: 15px; border-radius: 5px; margin: 20px 0;">
                <strong>Error Details:</strong><br>
                {{ error_message }}
            </div>
            
            {% if client_id %}
            <p><strong>Client ID:</strong> {{ client_id }}</p>
            {% endif %}
            
            <p><strong>Timestamp:</strong> {{ timestamp }}</p>
            
            <p>Please investigate and resolve this issue as soon as possible.</p>
            
            <p>Video Insights Monitoring System</p>
        </body>
        </html>
        """
        
        template = Template(html_template)
        html_content = template.render(
            error_message=error_message,
            client_id=client_id,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
        
        return self.send_email(
            to_emails=admin_emails,
            subject=subject,
            html_content=html_content
        )


# Global email service instance
email_service = EmailService()