from ttts.gpt.text import chinese, japanese, english, cleaned_text_to_sequence


language_module_map = {"ZH": chinese, "JP": japanese, "EN": english}


def clean_text1(text, language):
    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text)
    words = language_module.g2w(norm_text)
    return norm_text, words


def clean_text(text, language):
    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text)
    phones = language_module.g2p(norm_text)
    return norm_text, phones


def text_normalize(text, language):
    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text)
    return norm_text


def text_to_sequence(text, language):
    norm_text, phones = clean_text(text, language)
    return cleaned_text_to_sequence(phones, language)


if __name__ == "__main__":
    pass
