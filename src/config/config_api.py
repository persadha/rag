"""API-backed model providers for the r3 evaluation runs.

Generators (the LLM inside each RAG graph) come from hosted APIs so the whole
evaluation runs on a CPU-only machine — no GPU, no Ollama:
  - "haiku":      claude-haiku-4-5 via the Anthropic API (proprietary run)
  - "llama-groq": Llama 3.1 8B Instruct via Groq's OpenAI-compatible API (open-source run)

The DeepEval judge is configured in scripts/run_eval.py. Default is gpt-4.1
(OpenAI); set JUDGE_MODEL="compat:openai/gpt-oss-120b" in .env to use an
OpenAI-compatible OSS host (JUDGE_BASE_URL + JUDGE_API_KEY) instead.
Keys load from .env at the repo root (gitignored; see .env.example).

Every generator is piped through StrOutputParser so the graph nodes receive
plain strings, exactly as they do from OllamaLLM.
"""

import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


class APIConfig:
    ANTHROPIC_GENERATOR_MODEL = os.getenv("ANTHROPIC_GENERATOR_MODEL", "claude-haiku-4-5")
    OPENSOURCE_GENERATOR_MODEL = os.getenv("OPENSOURCE_GENERATOR_MODEL", "llama-3.1-8b-instant")
    OPENSOURCE_BASE_URL = os.getenv("OPENSOURCE_BASE_URL", "https://api.groq.com/openai/v1")
    JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4.1")  # "compat:openai/gpt-oss-120b" -> OSS judge
    JUDGE_BASE_URL = os.getenv("JUDGE_BASE_URL", "https://api.deepinfra.com/v1/openai")
    # JUDGE_API_KEY is read directly in run_eval.make_judge()

    # DeepInfra-hosted open-source generators (OpenAI-compatible). Reuses the DeepInfra
    # key (DEEPINFRA_API_KEY, falling back to JUDGE_API_KEY since the judge uses DeepInfra too).
    DEEPINFRA_BASE_URL = os.getenv("DEEPINFRA_BASE_URL", "https://api.deepinfra.com/v1/openai")
    GEMMA_DEEPINFRA_MODEL = os.getenv("GEMMA_DEEPINFRA_MODEL", "google/gemma-3-4b-it")

    # Closed-model comparison (E4): GPT-5.4-mini via OpenAI. Reasoning model — no temperature,
    # reasoning_effort kept low for a fair vs. non-reasoning open models comparison + lower cost.
    OPENAI_MINI_MODEL = os.getenv("OPENAI_MINI_MODEL", "gpt-5.4-mini")
    OPENAI_MINI_REASONING = os.getenv("OPENAI_MINI_REASONING", "low")

    # Local Ollama (no API key; used for key-less pilots and offline runs)
    OLLAMA_GENERATOR_MODEL = os.getenv("OLLAMA_GENERATOR_MODEL", "llama3:8b")
    OLLAMA_SLM_MODEL = os.getenv("OLLAMA_SLM_MODEL", "gemma3:1b")

    GENERATOR_NAMES = ("haiku", "llama-groq", "ollama", "gemma-deepinfra", "openai-mini")

    # Model id shown in the gen CSV per generator (for the local "ollama" path the
    # actual model comes from OLLAMA_GENERATOR_MODEL, set via env).
    @staticmethod
    def model_id_for(name: str) -> str:
        return {"haiku": APIConfig.ANTHROPIC_GENERATOR_MODEL,
                "llama-groq": APIConfig.OPENSOURCE_GENERATOR_MODEL,
                "ollama": APIConfig.OLLAMA_GENERATOR_MODEL,
                "gemma-deepinfra": APIConfig.GEMMA_DEEPINFRA_MODEL,
                "openai-mini": APIConfig.OPENAI_MINI_MODEL}.get(name, name)

    @staticmethod
    def get_generator(name: str):
        """Return a string-output runnable usable as both llm and slm in the graphs."""
        if name == "haiku":
            from langchain_anthropic import ChatAnthropic
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise RuntimeError("ANTHROPIC_API_KEY not set — copy .env.example to .env and fill it in")
            chat = ChatAnthropic(model=APIConfig.ANTHROPIC_GENERATOR_MODEL,
                                 temperature=0, max_tokens=1024, max_retries=5)
        elif name == "llama-groq":
            from langchain_openai import ChatOpenAI
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise RuntimeError("GROQ_API_KEY not set — copy .env.example to .env and fill it in")
            chat = ChatOpenAI(model=APIConfig.OPENSOURCE_GENERATOR_MODEL,
                              base_url=APIConfig.OPENSOURCE_BASE_URL, api_key=api_key,
                              temperature=0, max_tokens=1024, max_retries=5)
        elif name == "ollama":
            from langchain_ollama import ChatOllama
            from src.config.config import Config
            chat = ChatOllama(model=APIConfig.OLLAMA_GENERATOR_MODEL,
                              temperature=0, num_ctx=8192,
                              base_url=Config.OLLAMA_BASE_URL)
        elif name == "gemma-deepinfra":
            from langchain_openai import ChatOpenAI
            api_key = os.getenv("DEEPINFRA_API_KEY") or os.getenv("JUDGE_API_KEY")
            if not api_key:
                raise RuntimeError("DEEPINFRA_API_KEY (or JUDGE_API_KEY) not set for gemma-deepinfra")
            chat = ChatOpenAI(model=APIConfig.GEMMA_DEEPINFRA_MODEL,
                              base_url=APIConfig.DEEPINFRA_BASE_URL, api_key=api_key,
                              temperature=0, max_tokens=1024, max_retries=5)
        elif name == "openai-mini":
            from langchain_openai import ChatOpenAI
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY not set for openai-mini (E4 closed-model run)")
            # Reasoning model: omit temperature (only default supported); cap reasoning via effort.
            chat = ChatOpenAI(model=APIConfig.OPENAI_MINI_MODEL,
                              reasoning_effort=APIConfig.OPENAI_MINI_REASONING, max_retries=5)
        else:
            raise ValueError(f"Unknown generator '{name}'; expected one of {APIConfig.GENERATOR_NAMES}")
        return chat | StrOutputParser()

    @staticmethod
    def get_ollama_slm():
        """Small local grader (CRAG/CRAG++ design parity: gemma3:1b)."""
        from langchain_ollama import ChatOllama
        from src.config.config import Config
        chat = ChatOllama(model=APIConfig.OLLAMA_SLM_MODEL, temperature=0,
                          base_url=Config.OLLAMA_BASE_URL)
        return chat | StrOutputParser()
