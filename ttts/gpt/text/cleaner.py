from ttts.gpt.text import chinese, japanese, english, cleaned_text_to_sequence
import re


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


def split_sentences(text, min_len=15, max_len=80):
    punctuation = ["!", "?", ".", ";", "！", "？", "。", "；"]
    pattern = r"(?<=[{0}])\s*".format("".join(punctuation))
    sentences = [i for i in re.split(pattern, text) if i.strip() != ""]
    pattern = re.compile(
        r"([\u1100-\u11ff\u2e80-\ua4cf\ua840-\uD7AF\uF900-\uFAFF\uFE30-\uFE4F\uFF65-\uFFDC\U00020000-\U0002FFFF])")

    sen_lens = []
    merge_flags = []
    length = len(sentences)
    if length <= 1:
        return sentences

    for i in range(0, length):
        chars = pattern.split(sentences[i].strip())
        sen_len = 0
        for w in chars:
            sen_len += len(w.strip().split())
        sen_lens.append(sen_len)
        merge = True if sen_len > min_len else False
        merge_flags.append(merge)

    while not all(merge_flags):
        min_sen_len = min(sen_lens)
        idx = merge_flags.index(min_sen_len)
        if idx == 0:
            tag_idx = idx+1
        elif idx == length-1:
            tag_idx = idx-1
        else:
            tag_idx = idx-1 if sen_lens[idx-1] <= sen_lens[idx+1] else idx+1
        new_len = sen_lens[idx]+sen_lens[tag_idx]
        if new_len <= max_len:
            sen_lens[tag_idx] = new_len
            if new_len >= min_len:
                merge_flags[tag_idx] = True
            if tag_idx > idx:
                sentences[tag_idx] = sentences[idx]+" "+sentences[tag_idx]
            else:
                sentences[tag_idx] = sentences[tag_idx]+" "+sentences[idx]
            sen_lens.pop(idx)
            merge_flags.pop(idx)
            sentences.pop(idx)
        else:
            merge_flags[idx] = True
    return sentences


if __name__ == "__main__":
    pass
