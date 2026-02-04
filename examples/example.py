from text2tok import (
    reg_tokenize,
    reg_fast_tokenize,
    icu_tokenize,
    icu_fast_tokenize,
    BPETokenizer,
    BERTTokenizer,
)

text_list = [
    "去過中國science院，覺得it's pretty good。",
    "I'm having a state-of-the-art \"whopper\" at Mendy's and James'.",
    "I can’t ‘admire’ such a 'beautiful' dog.",
    "最多容納59,000個人,或5.9萬人,坪數對人數為1:3.",
]

tokenizer_list = [
    ("REG", reg_tokenize),
    ("ICU", icu_tokenize),
    ("REF", reg_fast_tokenize),
    ("ICF", icu_fast_tokenize),
    ("BPE", BPETokenizer("Qwen/Qwen3-8B")),
    ("BRT", BERTTokenizer("google-bert/bert-base-multilingual-cased")),
]

for text in text_list:
    print(f"{text}")
    for name, tokenize in tokenizer_list:
        token_list = tokenize(text)
        print(f"[{name}] {token_list}")
    print()
