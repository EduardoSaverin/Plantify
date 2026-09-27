import logging
import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
from transformers import AutoTokenizer, PreTrainedTokenizerBase

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3.1-8B-Instruct")

def tokenize(text: str):
    ids = tokenizer.encode(text=text, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(ids)
    decoded_text = tokenizer.decode(token_ids=ids, skip_special_tokens=True)

    logger.info(f"IDS[Size: {len(ids)}] : {ids}, Strings[Size: {len(tokens)}] : {tokens}")
    logger.info(f"Text :  {decoded_text}")

tokenize("plant")
tokenize(" plant")
tokenize("1000")
tokenize("1001")
tokenize("पौधा")
tokenize("What causes yellow leaves on a snake plant?")
