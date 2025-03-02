import logging
import os

# Define log directory (create if missing)
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Log file path
LOG_FILE_PATH = os.path.join(LOG_DIR, "tmdb_app.log")

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to INFO or WARNING in production
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),  # Save logs to file
        logging.StreamHandler()  # Also log to console
    ]
)

# Function to get module-specific loggers
def get_logger(name):
    return logging.getLogger(name)
