"""Application configuration loaded from environment."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    aws_region: str = "us-east-1"
    model_provider: str = "anthropic"
    tavily_api_key: str = ""
    fred_api_key: str = ""
    bedrock_kb_id: str = ""

    advisor_model: str = "arn:aws:bedrock:us-east-1:025066257437:application-inference-profile/whyo064oy367"
    client_model: str = "arn:aws:bedrock:us-east-1:025066257437:application-inference-profile/g51xi7ds9quj"
    analyst_model: str = "arn:aws:bedrock:us-east-1:025066257437:application-inference-profile/uz0tut3vgl7j"
    embedding_model: str = "arn:aws:bedrock:us-east-1:025066257437:application-inference-profile/saxnrtuaygnr"

    advisor_temperature: float = 0.2
    client_temperature: float = 0.7
    analyst_temperature: float = 0.0

    max_turns: int = 20
    knowledge_dir: str = "data/knowledge"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
