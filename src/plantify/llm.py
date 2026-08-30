import ollama
import logging
from plantify.config import settings
from functools import lru_cache

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def ollam_client():
    return ollama.Client(host=settings.ollama_host)

def _build_options(temperature = 0.0, seed=42, params=None) -> dict:
    if params is None:
        params = {}
    return {
        'temperature': temperature,
        'seed': seed,
        # 'num_predict': 150, # truncates the response in-between. For short response tell the LLM via prompt. completion_tokens=150 will become same
        **params
    }

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
    options = _build_options(temperature, seed, params)
    client = ollam_client()
    response = client.chat(model=model, messages=[
        {
            'role': 'system',
            'content': system,
        },
        {
            'role': 'user',
            'content': user,
        }
    ], options=options)
    _log_stats(response)
    return response.message.content

def stream_complete(system, user, model = None, temperature=0.0, seed=42, **params):
    model = model or settings.text_model
    options = _build_options(temperature, seed, params)
    client = ollam_client()
    response = client.chat(model=model, messages=[
        {
            'role': 'system',
            'content': system,
        },
        {
            'role': 'user',
            'content': user,
        }
    ], options=options, stream=True)
    final_chunk = None
    for chunk in response:
        final_chunk = chunk
        if chunk.message.content:
            yield chunk.message.content

    if final_chunk is not None:
        _log_stats(final_chunk)

if __name__ == '__main__':
    answer = complete(system="You are plants expert", user="What causes yellow leaves on a snake plant?")
    print(f"Answer : {answer}")

    for text in stream_complete(
            system="You are plants expert",
            user="What causes yellow leaves on a snake plant?"
    ):
        print(text, end="", flush=True)