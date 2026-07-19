import os


def _split_csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


class Settings:
    api_key: str = os.environ.get("QNA_API_KEY", "")
    hf_token: str = os.environ.get("HF_TOKEN", "")
    hf_qa_model: str = os.environ.get("HF_QA_MODEL", "deepset/roberta-base-squad2")
    allowed_origins: list[str] = _split_csv(os.environ.get("QNA_ALLOWED_ORIGINS", ""))
    allowed_hosts: list[str] = _split_csv(os.environ.get("QNA_ALLOWED_HOSTS", "*"))
    rate_limit: str = os.environ.get("QNA_RATE_LIMIT", "20/minute")
    max_question_length: int = int(os.environ.get("QNA_MAX_QUESTION_LENGTH", "300"))
    log_level: str = os.environ.get("QNA_LOG_LEVEL", "INFO")


settings = Settings()
