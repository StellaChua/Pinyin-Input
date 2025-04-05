import json
import sys
from collections import defaultdict
from collections import Counter
import math

POLYPHONIC_DICT = defaultdict(list)

# Load word to pinyin
def load_polyphonic_dict(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            char, pinyin = line.strip().split()
            POLYPHONIC_DICT[char].append(pinyin)

def is_char_match_pinyin(char, pinyin):
    if char not in POLYPHONIC_DICT:
        return True  
    return pinyin in POLYPHONIC_DICT[char]

# Load pinyin to word
def load_pinyin_to_chars(filename, max_candidates=600):
    pinyin_to_chars = defaultdict(list)
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for pinyin, info in data.items():
            chars = info['words']
            counts = info['counts']
            total = sum(counts)
            candidates = []
            for char, count in zip(chars, counts):
                try:
                    probability = math.log(count / total)
                except:
                    probability = -10000
                candidates.append((char, probability))
            # Sort the candidates by probability in descending order
            candidates.sort(key=lambda x: -x[1])
            candidates = candidates[:max_candidates]
            pinyin_to_chars[pinyin] = candidates
    return pinyin_to_chars

# Load double pinyin to words
def load_bigram_transitions(filename,max_transitions=600, alpha=1):
    joint_counts = Counter()
    prev_counts = Counter()
    
    with open(filename, 'r') as f:
        data = json.load(f)
        for bigram_pinyin, info in data.items():
            p1, p2 = bigram_pinyin.split() 
            for bigram, count in zip(info['words'], info['counts']):
                    if count >= 10:
                        c1, c2 = bigram.split()
                        joint_counts[(p1,c1,p2,c2)] += count
                        prev_counts[(p1,c1)] += count

    transitions = defaultdict(dict)
    vocab_size = len({c2 for (_,_,_,c2) in joint_counts}) 
    
    for (p1,c1,p2,c2), joint_count in joint_counts.items():
        prob = math.log(joint_count + alpha) - math.log(prev_counts[(p1,c1)] + alpha*vocab_size)
        transitions[(p1,c1)][(p2,c2)] = prob

    return {
        k: dict(sorted(v.items(), key=lambda x: -x[1])[:max_transitions])
        for k, v in transitions.items()
    }

def get_transition_prob(transitions, pinyin_to_chars, prev_pinyin, prev_char, curr_pinyin, curr_char):
    # Lookup transition prob
    bigram_prob = transitions.get((prev_pinyin, prev_char), {}).get((curr_pinyin, curr_char), math.log(1e-4))
    
    # Unigram prob of curr pinyin
    unigram_prob = math.log(1e-8)
    for char, prob in pinyin_to_chars.get(curr_pinyin, []):
        if char == curr_char:
            unigram_prob = prob
            if not is_char_match_pinyin(char, curr_pinyin):  
                penalty = -10
                unigram_prob += penalty # Punishment when word does not match pinyin
            break
    
    # Add weight ,
    weight = 0.88 if (prev_pinyin, prev_char) in transitions else 0.7
    return weight * bigram_prob + (1 - weight) * unigram_prob

def viterbi(pinyin_list, pinyin_to_chars, transitions):
    if not pinyin_list:
        return ""

    prev_dp = {}
    prev_path = {}
    
    # Handle first pinyin
    first_pinyin = pinyin_list[0]
    if first_pinyin not in pinyin_to_chars:
        return "" 
    
    for char, log_prob in pinyin_to_chars[first_pinyin]:
        if log_prob > -20 and is_char_match_pinyin(char, first_pinyin):
            prev_dp[char] = log_prob
            prev_path[char] = [char]

    # DP for the rest of the pinyin
    for i in range(1, len(pinyin_list)):
        current_pinyin = pinyin_list[i]
        prev_pinyin = pinyin_list[i-1]
        # Get the possible words and respective probs
        current_chars = pinyin_to_chars.get(current_pinyin, [])
        
        if not current_chars:
            return ""
        
        curr_dp = {}
        curr_path = {}
        
        for current_char, current_prob in current_chars:
            if not is_char_match_pinyin(current_char, current_pinyin):  # Check for polyphonic char
                continue
            max_log_prob = -float('inf')
            best_prev_path = None
            
            for prev_char, prev_log_prob in prev_dp.items():
                transition_prob = get_transition_prob(
                    transitions, pinyin_to_chars,
                    prev_pinyin, prev_char,
                    current_pinyin, current_char
                )
                
                total_log_prob = prev_log_prob + transition_prob
                if total_log_prob > max_log_prob:
                    max_log_prob = total_log_prob
                    best_prev_path = prev_path[prev_char]
            
            if max_log_prob > -float('inf'):
                curr_dp[current_char] = max_log_prob + math.log(max(current_prob, 1e-20))
                curr_path[current_char] = best_prev_path + [current_char]

            if len(curr_dp) > 50:
                top_items = sorted(curr_dp.items(), key=lambda x: -x[1])[:20]
                curr_dp = dict(top_items)
                curr_path = {k: curr_path[k] for k in curr_dp.keys()}

        prev_dp = curr_dp
        prev_path = curr_path

    if not prev_dp:
        return ""
    
    # Find the best path
    best_path = max(prev_path.values(), key=lambda path: prev_dp[path[-1]])
    return ''.join(best_path)

# Evaluate results
def evaluate(output_file, answer_file):
    with open(output_file, 'r', encoding='utf-8') as f1, open(answer_file, 'r', encoding='utf-8') as f2:
        output_lines = f1.read().strip().split("\n")
        answer_lines = f2.read().strip().split("\n")

    correct_chars = sum(sum(1 for o, a in zip(out, ans) if o == a) for out, ans in zip(output_lines, answer_lines))
    total_chars = sum(len(ans) for ans in answer_lines)
    correct_sentences = sum(out == ans for out, ans in zip(output_lines, answer_lines))

    print(f"Word Accuracy: {correct_chars / total_chars * 100:.2f}%", file=sys.stderr)
    print(f"Sentence Accuracy: {correct_sentences / len(answer_lines) * 100:.2f}%", file=sys.stderr)