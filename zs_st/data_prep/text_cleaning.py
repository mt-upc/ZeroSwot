import html
import re
from tqdm import tqdm

import nltk

from rpunct import RestorePuncts
from num2words import num2words


def restore_puncts(texts, bs=64):
    
    rpunct = RestorePuncts()
    
    lengths = [len(t.split(" ")) for t in texts]
    # sort by length and keep the index
    order = sorted(range(len(lengths)), key=lambda k: lengths[k], reverse=True)
    texts = [texts[i] for i in order]
    
    restored_texts = []
    pbar = tqdm(total=len(texts) // bs + 1)
    while texts:
        batch = []
        for _ in range(bs):
            if texts:
                batch.append(texts.pop())
        preds, _ = rpunct.model.predict(batch)
        for text, pred in zip(batch, preds):
            combined_pred = rpunct.combine_results(text, [pred])
            punct_text = rpunct.punctuate_texts(combined_pred)
            restored_texts.append(punct_text)

        pbar.update(1)
    pbar.close()
    
    restored_texts = restored_texts[::-1]
    restored_texts_ = [None]*len(restored_texts)
    for i, original_index in enumerate(order):
        restored_texts_[original_index] = restored_texts[i]

    return restored_texts_


def clean_speaker_name(text: str) -> str:
    """removes speaker name that might appear in the beginning of the sentence
    Args:
        text (str): text
    Returns:
        str: text without speaker name
    """
    if ": " in text:
        for sentence in nltk.sent_tokenize(text, language="english"):
            if ": " in sentence:
                start_text, rest_text = sentence.split(": ", maxsplit=1)
                start_tokens = re.sub(" +", " ", start_text).strip().split(" ")
                num_start_tokens = len(start_tokens)

                # XXX: one word, initials, all caps
                if num_start_tokens == 1 and start_tokens[0].isupper():
                    text = text.replace(sentence, rest_text)
                # Xxxx (Zzzz) Yyyy: two or three words, first (middle) last, start of each name is capitalized
                elif 1 < num_start_tokens < 4 and all(
                    [start_tokens[i][0].isupper() for i in range(num_start_tokens)]
                ):
                    text = text.replace(sentence, rest_text)

    return text


def clean_event(text: str) -> str:
    """removes event (e.g. laughter, applause) that appears enclosed in parenthesis
    Args:
        text (str): text
    Returns:
        str: text without events
    """

    # just parenthesis
    simple_event_pattern = r"\([^()]*\)"

    if ": " in text:
        for event in re.findall(simple_event_pattern, text):
            # check if event contains actual text from a speaker: (XX: utterance) -> utterance
            if ": " in event:
                event_text = event[1:-1]  # (xyz) -> xyz
                event_text_cleaned = clean_speaker_name(event_text)

                # replace event with its cleaned text
                if event_text != event_text_cleaned:
                    text = text.replace(event, event_text_cleaned)

    # remove rest of the events
    # parenthesis with punctuations, " . ... :, before or after
    all_event_patterns = r'"(\([^()]*\))"|"(\([^()]*\))|(\([^()]*\):)|(\([^()]*\)\.\.\.)|(\([^()]*\)\.)|(\([^()]*\))'
    text = re.sub(all_event_patterns, "", text)

    text = text.replace(" -- -- ", " -- ")

    return text


def replace_with_spelled_out_form(txt: str) -> str:
    txt = replace_numbers_with_words(txt)
    txt = txt.replace(" & ", " and ").replace("&", " and ")
    txt = txt.replace(" + ", " plus ").replace("+", " plus ")
    txt = txt.replace(" @ ", " at ").replace("@", " at ")
    txt = txt.replace(" % ", " percent ").replace("%", " percent ")
    txt = txt.replace(" = ", " equals ").replace("=", " equals ")
    txt = txt.replace(" $ ", " dollars ").replace("$", " dollars ")
    return txt

def replace_numbers_with_words(text):
    def convert_decades(match):
        number = match.group(0).strip("'")
        # print(number)
        if len(number) == 5 and number.endswith('0s'):  # Handle full decades like '1830s'
            return num2words(int(number[:4]), to='year') + "s"
        elif len(number) == 3:  # Handle decades like '90s'
            return num2words(int(number[:-1]), to='year') + 'ies'
        else:  # Handle centuries
            century = int(number[:2])
            return num2words(century, to='ordinal', lang='en') + " century"

    def convert_time(match):
        time_parts = match.group(0).split(':')
        hour = num2words(int(time_parts[0]), lang='en')
        minute = num2words(int(time_parts[1][:2]), lang='en') if int(time_parts[1][:2]) != 0 else ''
        period = time_parts[1][2:].strip().lower() if len(time_parts[1]) > 2 else ''
        time = hour + (' ' + minute if minute else '') + (' ' + period if period else '').strip()
        return time

    # Times
    text = re.sub(r'\b(?:[01]?[0-9]|2[0-3]):[0-5][0-9]\s?(AM|PM|A\.M\.|P\.M\.)?\b', convert_time, text)
    # Large numbers
    text = re.sub(r'\b\d{1,3}(?:,\d{3})*\b', lambda m: num2words(int(m.group(0).replace(',', '')), lang='en'), text)
    # Decades and years with 's'
    text = re.sub(r"\b\d{2,4}s\b", convert_decades, text)
    # Ordinal numbers (e.g., 21st, 400th)
    text = re.sub(r'\b\d+(st|nd|rd|th)\b', lambda m: num2words(int(m.group(0)[:-2]), to='ordinal', lang='en'), text)
    # All other numbers
    text = re.sub(r'\b\d+\b', lambda m: num2words(int(m.group(0)), lang='en'), text)

    return text


def remove_spaces(txt: str) -> str:
    txt = re.sub(" +", " ", txt.strip().replace("\t", " ").replace("\n", " "))
    return txt


def handle_html_non_utf(txt: str) -> str:
    """handles html and non-utf chars"""
    txt = html.unescape(bytes(txt, "utf-8").decode("utf-8", "ignore"))
    return txt

def normalize_punctuation(txt: str) -> str:
    replacement_dict = {
        '—': '-', 
        ' ’ ': "'",
        '–': '-',
        '„': '"',
        '“': '"',
        '»': '"',
        '«': '"',
        '”': '"',
        '’': "'",
        '‘': "'",
        '‚': "'",
        "[": "(",
        "]": ")",
        "{": "(",
        "}": ")",
    }
    for symbol, replacement in replacement_dict.items():
        txt = txt.replace(symbol, replacement)
    if txt.startswith('"') and txt.endswith('"') and txt.count('"') == 2:
        txt = txt[1:-1]
    if txt.count('"') == 1:
        txt = txt.replace('"', "")
    if "(" in txt and ")" not in txt:
        txt = txt.replace("(", "")
    if ")" in txt and "(" not in txt:
        txt = txt.replace(")", "")
    if not txt.endswith((".", "?", "!", ",", "-", ":", ";")):
        txt = txt + "."
    while "...." in txt:
        txt = txt.replace("....", "...")
    while "!!" in txt:
        txt = txt.replace("!!", "!")
    return txt


def clean_text(txt: str, is_mustc: bool) -> str:
    txt = remove_spaces(txt)
    txt = handle_html_non_utf(txt)
    if is_mustc:
        txt = clean_event(txt)
        txt = clean_speaker_name(txt)
    txt = normalize_punctuation(txt)
    txt = " ".join(txt.split())
    txt = txt.strip()
    if set(txt) == set(" "):
        return ""
    else:
        return txt


def tokenize_asr_text(cleaned_txt: str, vocab: list) -> str:
    """tokenizes the output of "clean_text" according to the wav2vec2.0 vocab
    Args:
        cleaned_txt (str): output of the clean_text functions
        vocab (list): wav2vec2.0 char vocab
    Returns:
        str: tokenized text tokens separated by |
    """

    cleaned_txt = cleaned_txt.upper()

    tokenized_cleaned_txt = "".join(
        [c if c in vocab else "|" for c in cleaned_txt.replace(" ", "|")]
    ) + "|"

    if set(tokenized_cleaned_txt) == set("|") or not tokenized_cleaned_txt:
        return ""

    while "||" in tokenized_cleaned_txt:
        tokenized_cleaned_txt = tokenized_cleaned_txt.replace("||", "|")
    if tokenized_cleaned_txt[0] == "|":
        tokenized_cleaned_txt = tokenized_cleaned_txt[1:]

    tokenized_cleaned_txt = " ".join(list(tokenized_cleaned_txt))

    if set(tokenized_cleaned_txt) == set(" "):
        return ""
    return tokenized_cleaned_txt


def tokenize_asr_text_punct(cleaned_txt: str) -> str:
    
    tokenized_cleaned_txt = cleaned_txt.upper().replace(" ", "|").replace("▁", "") + "|"
    tokenized_cleaned_txt = " ".join(list(tokenized_cleaned_txt))
    return tokenized_cleaned_txt