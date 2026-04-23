import os
import time
import subprocess
import datetime
import logging
import smtplib
from typing import List
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Try to load dotenv, otherwise manually parse .env
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, '.env')

def get_docker_compose_command():
    """Determines the correct Docker Compose command (v1 vs v2)."""
    try:
        # Check for 'docker compose' (v2 plugin)
        subprocess.run(["docker", "compose", "version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return ["docker", "compose"]
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    try:
        # Check for 'docker-compose' (standalone)
        subprocess.run(["docker-compose", "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return ["docker-compose"]
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Default fallback (user will see error if neither exists)
    logging.warning("No docker-compose command found/verified. Defaulting to 'docker compose'.")
    return ["docker", "compose"]

DOCKER_COMPOSE_CMD = get_docker_compose_command()


try:
    from dotenv import load_dotenv
    load_dotenv(env_path)
except ImportError:
    # Manual parsing of .env file
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

# --- Configuration ---
# Set the times to start and stop services (24-hour format HH:MM)
START_TIME = os.getenv("START_TIME", "09:00")
STOP_TIME = os.getenv("STOP_TIME", "22:00")

# Email Configuration
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")
# Parse multiple recipients (comma-separated), strip whitespace
_recipient_env = os.getenv("RECIPIENT_EMAIL", "")
RECIPIENT_EMAILS = [email.strip() for email in _recipient_env.split(',') if email.strip()]

# List of services to manage. 
# We explicitly list services here to EXCLUDE "unsafe" ones (MinIO, Redis) that lack volumes.
SERVICES_TO_MANAGE: List[str] = [
    "mongodb",
    "loki",
    "grafana",
    "counter_staff_detection_service1",
    "cafe_counter_staff_detection_service",
    "table_occupancy_back_lobby",
    "table_occupancy_left_lobby",
    "table_occupancy_cafe",
    "table_occupancy_kiosk",
    "entry_exit_service",
    "exit_emotion_detection_service",
    "crew_room_service",
    "kiosk_person_service",
    "db_service",
    "db_sync_service",
    "grounding_dino_service",
    "stream_handling_service",
    "yolo_service",
    "staff_customer_classification_service",
    "ip_monitor_service"
]

# Path to your docker-compose file dir
WORKING_DIR = os.getenv("BACKEND_PROJECT_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 

# --- Setup Logging ---
log_file = os.path.join(WORKING_DIR, "scheduler.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

def send_email(subject, body):
    """Sends an email notification to multiple recipients."""
    if not SENDER_EMAIL or not SENDER_PASSWORD or not RECIPIENT_EMAILS:
        logging.warning("Email configuration missing or no recipients. Skipping email notification.")
        return

    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        # Join emails with comma for the header
        msg['To'] = ", ".join(RECIPIENT_EMAILS)
        msg['Subject'] = subject

        # Use HTML for better formatting
        msg.attach(MIMEText(body, 'html'))

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        
        # Send to the list of recipients
        server.sendmail(SENDER_EMAIL, RECIPIENT_EMAILS, msg.as_string())
        server.quit()
        
        logging.info(f"📧 Email sent to: {', '.join(RECIPIENT_EMAILS)} | Subject: {subject}")
    except Exception as e:
        logging.error(f"❌ Failed to send email: {e}")

def run_command(command_list, cwd=WORKING_DIR):
    """Executes a shell command and logs output."""
    try:
        logging.info(f"Executing: {' '.join(command_list)}")
        result = subprocess.run(
            command_list, 
            cwd=cwd, 
            capture_output=True, 
            text=True, 
            check=True
        )
        logging.info(f"Output: {result.stdout}")
        if result.stderr:
            logging.warning(f"Stderr: {result.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed: {e}")
        logging.error(f"Error output: {e.stderr}")
        return False

def get_current_time_str():
    return datetime.datetime.now().strftime("%H:%M")

def start_services():
    logging.info("🌞 It is Morning! Starting services...")
    success = False
    if not SERVICES_TO_MANAGE:
        # Full stack up
        success = run_command(DOCKER_COMPOSE_CMD + ["up", "-d"])
        target_desc = "<b>ALL services</b> (Full Stack)"
    else:
        # Specific services start
        success = run_command(DOCKER_COMPOSE_CMD + ["start"] + SERVICES_TO_MANAGE)
        service_list = "".join([f"<li>{s}</li>" for s in SERVICES_TO_MANAGE])
        target_desc = f"<b>Selected Services:</b><ul>{service_list}</ul>"
    
    time_str = get_current_time_str()
    if success:
        send_email(
            f"Services Started ({time_str})", 
            f"""
            <h2>🌞 Services Started Successfully</h2>
            <p><b>Time:</b> {time_str}</p>
            <p>The uptime scheduler has successfully started the following services:</p>
            {target_desc}
            <p><i>System is now active.</i></p>
            """
        )
    else:
        send_email(
            f"❌ START FAILED ({time_str})", 
            f"""
            <h2>❌ Service Start Failed</h2>
            <p><b>Time:</b> {time_str}</p>
            <p>The uptime scheduler attempted to start the services but encountered errors.</p>
            {target_desc}
            <p style="color:red;"><b>Please check the logs for details.</b></p>
            """
        )

def stop_services():
    logging.info("🌙 It is Night! Stopping services...")
    success = False
    if not SERVICES_TO_MANAGE:
        # Full stack down
        success = run_command(DOCKER_COMPOSE_CMD + ["down"])
        target_desc = "<b>ALL services</b> (Full Stack)"
    else:
        # Specific services stop
        success = run_command(DOCKER_COMPOSE_CMD + ["stop"] + SERVICES_TO_MANAGE)
        service_list = "".join([f"<li>{s}</li>" for s in SERVICES_TO_MANAGE])
        target_desc = f"<b>Selected Services:</b><ul>{service_list}</ul>"

    time_str = get_current_time_str()
    if success:
        send_email(
            f"Services Stopped ({time_str})", 
            f"""
            <h2>🌙 Services Stopped Successfully</h2>
            <p><b>Time:</b> {time_str}</p>
            <p>The uptime scheduler has successfully stopped the following services:</p>
            {target_desc}
            <p><i>System is now resting.</i></p>
            """
        )
    else:
        send_email(
            f"❌ STOP FAILED ({time_str})", 
            f"""
            <h2>❌ Service Stop Failed</h2>
            <p><b>Time:</b> {time_str}</p>
            <p>The uptime scheduler attempted to stop the services but encountered errors.</p>
            {target_desc}
            <p style="color:red;"><b>Please check the logs for details.</b></p>
            """
        )

def main():
    logging.info(f"🚀 Uptime Scheduler Started.")
    logging.info(f"   Working Directory: {WORKING_DIR}")
    logging.info(f"   Start Time: {START_TIME}")
    logging.info(f"   Stop Time: {STOP_TIME}")
    logging.info(f"   Services: {SERVICES_TO_MANAGE if SERVICES_TO_MANAGE else 'ALL (Full Stack)'}")
    
    if SENDER_EMAIL:
        logging.info("   📧 Email Notifications: ENABLED")
    else:
        logging.info("   ⚠️ Email Notifications: DISABLED (Missing Credentials)")

    last_action = None 
    
    logging.info("Waiting for time triggers...")

    while True:
        current_time = get_current_time_str()
        
        if current_time == START_TIME:
            if last_action != "start":
                start_services()
                last_action = "start"
                time.sleep(60) 
                
        elif current_time == STOP_TIME:
            if last_action != "stop":
                stop_services()
                last_action = "stop"
                time.sleep(60)

        # Heartbeat logic could go here
        
        time.sleep(30)

if __name__ == "__main__":
    main()
