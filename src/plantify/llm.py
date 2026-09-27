import ollama
import logging
from plantify.config import settings
from functools import lru_cache

logger = logging.getLogger(__name__)

_WARN_RATIO = 0.7

@lru_cache(maxsize=1)
def _tokenizer():
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(settings.tokenizer_model, local_files_only=True)
        return tokenizer
    except Exception as e:
        logger.debug("Error in initializing tokenizer")
        return None

@lru_cache(maxsize=1)
def ollam_client():
    return ollama.Client(host=settings.ollama_host)

def _build_options(temperature = 0.0, seed=42, num_ctx = None, params=None) -> dict:
    num_ctx = num_ctx or settings.num_ctx
    if params is None:
        params = {}
    return {
        'temperature': temperature,
        'seed': seed,
        'num_ctx': num_ctx,
        # 'num_predict': 150, # truncates the response in-between. For short response tell the LLM via prompt. completion_tokens=150 will become same
        **params
    }

def _build_message(system: str, user: str) -> list[dict]:
    return [
        {
            'role': 'system',
            'content': system,
        },
        {
            'role': 'user',
            'content': user,
        }
    ]

def _warn_if_over_budget(messages: list[dict], num_ctx: int) -> None:
    token_count = sum(count_tokens(message['content']) for message in messages)
    if token_count >= num_ctx*_WARN_RATIO:
        logger.warning(f"Prompt tokens {token_count}, {token_count / num_ctx * 100:.2f}% of num_ctx")

def _log_stats(final_chunk) -> None:
    elapsed_seconds = final_chunk.total_duration / 1_000_000_000
    prompt_tokens = final_chunk.prompt_eval_count
    completion_tokens = final_chunk.eval_count

    tps = (
        completion_tokens / (final_chunk.eval_duration / 1e9)
        if final_chunk.eval_duration
        else 0
    )

    load_seconds = (
        final_chunk.load_duration / 1e9
        if final_chunk.load_duration
        else 0
    )
    logger.info(
        "\nOllama Response. model=%s, prompt_tokens=%d, "
        "completion_tokens=%d, elapsed=%.2fs, load=%.2fs, tps=%.2ftok/s",
        final_chunk.model,
        prompt_tokens,
        completion_tokens,
        elapsed_seconds,
        load_seconds,
        tps
    )

def complete(system, user, model = None, temperature=0.0, seed=42, **params):
    model = model or settings.text_model
    options = _build_options(temperature, seed, settings.num_ctx, params)
    client = ollam_client()
    message = _build_message(system, user)
    _warn_if_over_budget(message, settings.num_ctx)
    response = client.chat(model=model, messages=message, options=options)
    _log_stats(response)
    return response.message.content

def stream_complete(system, user, model = None, temperature=0.0, seed=42, **params):
    model = model or settings.text_model
    options = _build_options(temperature, seed, settings.num_ctx, params)
    client = ollam_client()
    message = _build_message(system, user)
    _warn_if_over_budget(message, settings.num_ctx)
    response = client.chat(model=model, messages=message, options=options, stream=True)
    final_chunk = None
    for chunk in response:
        final_chunk = chunk
        if chunk.message.content:
            yield chunk.message.content

    if final_chunk is not None:
        _log_stats(final_chunk)

def count_tokens(text: str) -> int:
    tokenizer = _tokenizer()
    if tokenizer is None:
        return len(text.encode("utf-8")) // 4
    else:
        try:
            encoded = tokenizer.encode(text, add_special_tokens=False)
            return len(encoded)
        except Exception as e:
            return len(text.encode("utf-8")) // 4


if __name__ == '__main__':
    answer = complete(system="You are plants expert", user="What causes yellow leaves on a snake plant?")
    print(f"Answer : {answer}")

    for text in stream_complete(
            system="You are plants expert",
            user="What causes yellow leaves on a snake plant?"
    ):
        print(text, end="", flush=True)