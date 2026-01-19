# Device name mapping (keep existing configuration)
DeviceName = {
    "73f95560": "PDCN",
}
# Device timeout configuration (seconds)
DEVICE_TIMEOUT = 30  # Device considered offline after 30 seconds of inactivity
SCHEDULES_FILE = "schedules.json"
# Authentication related configuration
SYSTEM_PASSWORD = "admin123"  # Should be read from environment variables or config file in production


import logging
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)