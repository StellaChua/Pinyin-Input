import sys
import os
from src.viterbi import viterbi, load_polyphonic_dict, load_pinyin_to_chars, load_bigram_transitions, evaluate

def main():
    # Load data from fixed files
    load_polyphonic_dict('./data/word2pinyin.txt')
    pinyin_to_chars = load_pinyin_to_chars('./processed_data/1_word_prob.txt')
    transitions = load_bigram_transitions('./processed_data/2_word_prob.txt')
 
    if sys.stdin.isatty():
        input_lines = open('./data/input.txt', 'r', encoding='utf-8', file=sys.stderr)
    else:
        input_lines = sys.stdin

    if sys.stdout.isatty():
        output_file = open('./data/output.txt', 'w', encoding='utf-8', file=sys.stderr)
    else:
        output_file = sys.stdout

    for line in input_lines:
        pinyin_list = line.strip().split()
        result = viterbi(pinyin_list, pinyin_to_chars, transitions)
        output_file.write(result + '\n')

    evaluate('./data/output.txt', './data/answer.txt')

if __name__ == '__main__':
    main()