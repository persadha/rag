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

    # Local Ollama (no API key; used for key-less pilots and offline runs)
    OLLAMA_GENERATOR_MODEL = os.getenv("OLLAMA_GENERATOR_MODEL", "llama3:8b")
    OLLAMA_SLM_MODEL = os.getenv("OLLAMA_SLM_MODEL", "gemma3:1b")

    GENERATOR_NAMES = ("haiku", "llama-groq", "ollama")

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
            chat = ChatOllama(model=APIConfig.OLLAMA_GENERATOR_MODEL,
                              temperature=0, num_ctx=8192)
        else:
            raise ValueError(f"Unknown generator '{name}'; expected one of {APIConfig.GENERATOR_NAMES}")
        return chat | StrOutputParser()

    @staticmethod
    def get_ollama_slm():
        """Small local grader (CRAG/CRAG++ design parity: gemma3:1b)."""
        from langchain_ollama import ChatOllama
        chat = ChatOllama(model=APIConfig.OLLAMA_SLM_MODEL, temperature=0)
        return chat | StrOutputParser()
