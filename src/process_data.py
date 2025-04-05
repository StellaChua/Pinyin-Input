import json
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from pypinyin import pinyin, Style

def is_chinese_char(char: str) -> bool:
    return '\u4e00' <= char <= '\u9fff'

def clean_text(text: str) -> str:
    allowed_punctuation = {'，', '。', '？', '！', '、', '；', '：', '（', '）', '「', '」', '『', '』'}
    return ''.join([char for char in text.strip() if is_chinese_char(char) or char in allowed_punctuation])

def preprocess_corpus(corpus_dir: Path, min_count: int = 0) -> tuple:
    unigram_counts = defaultdict(int)
    bigram_counts = defaultdict(int)
    file_count = 0

    txt_files = list(corpus_dir.glob('**/*.txt'))
    
    for file_path in tqdm(txt_files, desc="Processing files"):
        try:
            with open(file_path, 'r', encoding='gbk', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        text = data.get('html', data.get('title', ''))
                    except json.JSONDecodeError:
                        # Use the original line directly
                        text = line
                    cleaned_text = clean_text(text)
                    if not cleaned_text:
                        continue
                    chars = list(cleaned_text)
                    for char in chars:
                        unigram_counts[char] += 1
                    for i in range(len(chars) - 1):
                        bigram = (chars[i], chars[i+1])
                        bigram_counts[bigram] += 1
            file_count += 1
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue

    unigram_counts = {k: v for k, v in unigram_counts.items() if v >= min_count}
    bigram_counts = {k: v for k, v in bigram_counts.items() if v >= min_count}
    
    print(f"Processed {file_count} files. Found {len(unigram_counts)} unigrams and {len(bigram_counts)} bigrams.")
    return unigram_counts, bigram_counts

def generate_pinyin_to_chars(unigram_counts: dict) -> dict:
    # Mapping of words and counts for pinyin
    pinyin_to_chars = defaultdict(lambda: {"words": [], "counts": []})
    for char, count in unigram_counts.items():
        char_pinyin = pinyin(char, style=Style.NORMAL, heteronym=False)
        if char_pinyin:
            pinyin_str = char_pinyin[0][0]
            pinyin_to_chars[pinyin_str]["words"].append(char)
            pinyin_to_chars[pinyin_str]["counts"].append(count)

    for pinyin_str, info in pinyin_to_chars.items():
        sorted_candidates = sorted(zip(info["words"], info["counts"]), key=lambda x: -x[1])
        pinyin_to_chars[pinyin_str]["words"], pinyin_to_chars[pinyin_str]["counts"] = map(list, zip(*sorted_candidates))
    return pinyin_to_chars

def save_pinyin_to_chars(pinyin_to_chars: dict, output_file: Path):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(pinyin_to_chars, f, ensure_ascii=False, indent=2)

def save_bigram_counts(bigram_counts: dict, output_file: Path):
    # Mapping of words and counts for binary pinyin
    bigram_pinyin_groups = defaultdict(lambda: {"words": [], "counts": []})
    for (c1, c2), count in bigram_counts.items():
        p1 = pinyin(c1, style=Style.NORMAL, heteronym=False)[0][0]
        p2 = pinyin(c2, style=Style.NORMAL, heteronym=False)[0][0]
        key = f"{p1} {p2}"
        word = f"{c1} {c2}"
        bigram_pinyin_groups[key]["words"].append(word)
        bigram_pinyin_groups[key]["counts"].append(count)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(bigram_pinyin_groups, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    corpus_path = Path("./corpus")
    output_path = Path("./processed_data")
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Clean corpus
    unigrams, bigrams = preprocess_corpus(corpus_path)
    
    pinyin_to_chars = generate_pinyin_to_chars(unigrams)
    save_pinyin_to_chars(pinyin_to_chars, output_path / "1_word_prob.txt")

    save_bigram_counts(bigrams, output_path / "2_word_prob.txt")