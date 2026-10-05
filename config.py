"""Central paths and settings for the whole project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

RAW_DATA_PATH = ROOT / "data" / "sales_data.xlsx"      # input sheet (swap in your own!)
OUTPUT_DIR = ROOT / "outputs"
ARTIFACTS_DIR = OUTPUT_DIR / "artifacts"                 # files the chatbot reads
REPORT_PATH = OUTPUT_DIR / "sales_report.xlsx"          # final Excel report

# Forecasting
WINDOW = 6            # months of history the model looks at
EPOCHS = 600
LEARNING_RATE = 0.01
HOLDOUT_MONTHS = 3    # months kept aside to measure forecast accuracy

# Segmentation
K_RANGE = range(3, 7)  # KMeans tries k = 3..6 and keeps the best silhouette score

# Chatbot LLM (any LangChain "provider:model" string)
#   Free options:  "groq:llama-3.3-70b-versatile"  (needs GROQ_API_KEY, free tier)
#                  "ollama:llama3.1"               (runs locally, needs Ollama installed)
LLM_MODEL = "groq:llama-3.3-70b-versatile"
