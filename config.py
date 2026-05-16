"""Configuration for Custom Vision API access."""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load from environment variables for security
ENDPOINT = os.environ.get("ENDPOINT", "https://eastus.api.cognitive.microsoft.com/")
TRAINING_KEY = os.environ.get("CUSTOM_VISION_KEY", "")

# Output directory for downloaded data
OUTPUT_DIR = "custom_vision_export"

# Validate configuration
if not TRAINING_KEY:
    print("Warning: CUSTOM_VISION_KEY environment variable not set.")
    print("Set it with: export CUSTOM_VISION_KEY='your-api-key'")
