"""
Configuration settings for the CT Recommendation Agent.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Application configuration."""
    
    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o")
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
    
    # LangSmith Configuration (Optional - for tracing/debugging)
    LANGCHAIN_TRACING_V2: bool = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    LANGCHAIN_API_KEY: Optional[str] = os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "ct-recommendation-agent")
    
    # Application Settings
    MAX_CONVERSATION_HISTORY: int = 20
    MAX_TOKENS: int = 4000
    
    # Streamlit Settings
    PAGE_TITLE: str = "CT Scan Health Advisor"
    PAGE_ICON: str = "🏥"
    LAYOUT: str = "wide"
    
    @classmethod
    def validate(cls) -> bool:
        """Validate required configuration."""
        if not cls.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is not set. Please add it to your .env file."
            )
        return True
    
    @classmethod
    def get_model_config(cls) -> dict:
        """Get model configuration as dictionary."""
        return {
            "model_name": cls.MODEL_NAME,
            "temperature": cls.TEMPERATURE,
            "api_key": cls.OPENAI_API_KEY
        }


# Validate configuration on import
try:
    Config.validate()
except ValueError as e:
    print(f"⚠️ Configuration Warning: {e}")
    print("Please copy .env.example to .env and add your OpenAI API key.")
