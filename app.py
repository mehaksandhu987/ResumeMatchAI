"""Entry point for Streamlit Community Cloud / Hugging Face Spaces."""
from pathlib import Path
import runpy

# Run the main Streamlit app
runpy.run_path(str(Path(__file__).parent / "app" / "streamlit_app.py"), run_name="__main__")
