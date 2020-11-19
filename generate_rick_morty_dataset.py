import argparse
import json
import re

import numpy as np
import pandas as pd


RICK_IDENTITY = [
    'i am richard "rick" sanchez .',
    'i am also known as rick c-137 .',
]
RICK_PERSONALITY = [
    'i am a genius scientist, capable of creating complex scientific inventions .',
    'i used to be a rock star .',
    'i love to drink .',
    'i invented portals to several different dimensions, various energy weapons and force fields .',
    'i concerned about my daughter beth .',
    'i am caring about morty .',
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-src', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--full', action='store_true')

    return parser.parse_args()


def preprocess(s: str):
    s = s.lower()
    s = re.sub(r"([^ ])(\.)", r"\1 \2", s)
    s = re.sub(r"(\.)([^ ])", r"\1 \2", s)
    return s


def main():
    args = parse_args()
    all_rick = pd.read_csv(args.csv_src)

    contexted = []
    n = 7

    for i in range(n, len(all_rick['line'])):
        row = []
        prev = i - 1 - n  # we additionally substract 1, so row will contain current responce and 7 previous responces
        # Skip non-Rick replies
        if not args.full and 'Rick' not in all_rick['name'][i]:
            continue

        for j in range(i, prev, -1):
            to_append = preprocess(all_rick['line'][j])
            row.append(to_append)
        contexted.append(row)

    columns = ['response', 'context']
    columns = columns + ['context/' + str(i) for i in range(n - 1)]
    df = pd.DataFrame.from_records(contexted, columns=columns)

    replies = all_rick['line']
    entries = []

    for i, row in df.iterrows():
        history = list(reversed([rep for rep in row]))
        answer = history[-1]
        history_1 = history[:-1]
        candidates = np.random.choice(replies, size=20, replace=False).tolist() + [answer]
        personality = RICK_IDENTITY + np.random.choice(RICK_PERSONALITY, size=2, replace=False).tolist()

        entry = {
            'personality': personality,
            'utterances': [
                {
                    'candidates': candidates,
                    'history': history_1
                }
            ],
        }
        if i == len(df) - 1:
            break
        entries.append(entry)

        if int(i) % 200 == 0:
            print(f'Processed {i} rows.')

    dataset = {'train': entries, 'valid': [entry]}
    with open(args.output, 'w') as f:
        f.write(json.dumps(dataset))

    print(f'Saved to {args.output}.')


if __name__ == '__main__':
    main()
