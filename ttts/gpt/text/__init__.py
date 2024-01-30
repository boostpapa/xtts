from ttts.gpt.text.symbols import *
from .zh_normalization import *

_symbol_to_id = {s: i for i, s in enumerate(symbols)}


def cleaned_text_to_sequence(cleaned_text, language):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      cleaned_text: string to convert to a sequence
      language: language type
    Returns:
      List of integers corresponding to the symbols in the text
    """
    phones = [_symbol_to_id[symbol] for symbol in cleaned_text]
    lang_id = language_id_map[language]
    lang_ids = [lang_id for i in phones]
    return phones, lang_ids
