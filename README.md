# @palindrom-ai/llm

Unified Python LLM SDK for Palindrom services.

## Installation

```bash
pip install palindrom-ai-llm
```

For development:

```bash
pip install -e ".[dev]"
```

## Features

- **Unified API**: Single interface for OpenAI, Anthropic, and Google models
- **Structured Output**: All LLM calls return validated Pydantic models
- **Simple RAG**: Vector search with ChromaDB
- **Observability**: Built-in Langfuse integration
- **Evaluation**: DeepEval integration for testing

## Quick Start

```python
from palindrom_ai.llm import complete, extract
from pydantic import BaseModel

# Simple completion
response = await complete(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
)

# Structured extraction
class User(BaseModel):
    name: str
    age: int

user = await extract(
    response_model=User,
    model="gpt-4o",
    prompt="John is 25 years old",
)
print(user.name)  # "John"
```

## Documentation

- [Research Docs](./docs/research/)
- [Implementation Plans](./docs/plans/)

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run linting
ruff check .

# Run type checking
mypy src/

# Run tests
pytest
```
