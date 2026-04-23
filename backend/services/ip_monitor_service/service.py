import requests
import smtplib
import time
import os
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("IPMonitor")

# Load environment variables
load_dotenv()

SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")
RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL")
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", 300))  # Default 5 minutes
MIN_EMAIL_INTERVAL = int(os.getenv("MIN_EMAIL_INTERVAL", 3600))  # Default 1 hour cooldown
IP_STORE_FILE = "last_ip.txt"

# Rate limiting state
last_email_time = 0
emails_sent_today = 0
last_reset_day = time.strftime('%d')

def get_public_ip():
    """Fetches IP from multiple sources and returns consensus."""
    sources = [
        'https://api.ipify.org',
        'https://ifconfig.me/ip',
        'https://icanhazip.com'
    ]
    results = []
    for source in sources:
        try:
            response = requests.get(source, timeout=10)
            if response.status_code == 200:
                results.append(response.text.strip())
        except Exception as e:
            logger.warning(f"Source {source} failed: {e}")
    
    if not results:
        return None
        
    # Consensus logic: Return the most common result
    return max(set(results), key=results.count)

def send_email(old_ip, new_ip):
    global last_email_time, emails_sent_today, last_reset_day
    
    # Rate Limit Check: Daily Cap (5 emails)
    current_day = time.strftime('%d')
    if current_day != last_reset_day:
        emails_sent_today = 0
        last_reset_day = current_day
        
    if emails_sent_today >= 5:
        logger.warning("Daily email limit reached (5). Suppression active.")
        return

    # Rate Limit Check: Cooldown (1 hour)
    current_time = time.time()
    if (current_time - last_email_time) < MIN_EMAIL_INTERVAL:
        logger.warning(f"Email suppressed. Last email sent less than {MIN_EMAIL_INTERVAL}s ago.")
        return

    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECIPIENT_EMAIL
        msg['Subject'] = f"URGENT: Public IP Changed - {new_ip}"

        body = f"""
        Public IP address change detected and verified.
        
        Old IP: {old_ip if old_ip else 'Unknown/None'}
        New IP: {new_ip}
        
        Action Required: Update MongoDB Whitelist/Security Group rules.
        
        Detected at: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}
        Service status: Monitoring active.
        """
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()
        
        last_email_time = current_time
        emails_sent_today += 1
        logger.info(f"Verified Change: Email sent to {RECIPIENT_EMAIL}")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")

def main():
    logger.info("Hardened IP Monitoring Service Started")
    
    last_ip = None
    if os.path.exists(IP_STORE_FILE):
        with open(IP_STORE_FILE, 'r') as f:
            last_ip = f.read().strip()
    
    logger.info(f"Last recorded IP: {last_ip}")

    # For verification stability
    pending_ip = None
    verification_count = 0

    while True:
        current_ip = get_public_ip()
        
        if current_ip:
            if current_ip != last_ip:
                # Stability Check: Ensure IP is stable for 2 consecutive polls
                if current_ip == pending_ip:
                    verification_count += 1
                else:
                    pending_ip = current_ip
                    verification_count = 1
                
                if verification_count >= 2:
                    logger.info(f"IP Change Verified: {last_ip} -> {current_ip}")
                    send_email(last_ip, current_ip)
                    
                    last_ip = current_ip
                    with open(IP_STORE_FILE, 'w') as f:
                        f.write(current_ip)
                    verification_count = 0
                    pending_ip = None
                else:
                    logger.info(f"IP change detected, waiting for verification... (Poll {verification_count})")
            else:
                # Reset verification if IP returns to normal
                verification_count = 0
                pending_ip = None
                logger.debug(f"IP stable: {current_ip}")
        
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
