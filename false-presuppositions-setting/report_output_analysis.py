#!/usr/bin/env python3
"""Generate a text report from a model output folder.

Usage: python report_output_analysis.py /path/to/output/folder

The script looks for base.csv, critical.csv, presupposition.csv and prints
category counts and sample cleaned examples.
"""
import argparse
import csv
import glob
import os
import re
import html
from collections import Counter, defaultdict


BLOCKLIST_TOKENS = [
    "you are a helpful assistant",
    "gunta",
    "comey",
    "libertine",
    "system",
]


def first_sentence(text):
    if not text:
        return ""
    # normalize newlines
    t = text.strip().replace('\r', '\n')
    # If there's a clear first-line greeting, prefer that trimmed line
    first_line = t.split('\n', 1)[0].strip()
    # split into sentences by ., ?, ! followed by space or end
    m = re.split(r'(?<=[\.\?!])\s+', first_line)
    if m and m[0].strip():
        return m[0].strip()
    # fallback: use first 120 chars
    return (first_line[:120]).strip()


def is_injected(text):
    low = text.lower()
    for token in BLOCKLIST_TOKENS:
        if token in low:
            return True
    # repeated special markers: many non-alpha tokens or suspicious capitals
    if re.search(r"(\bcome+y\b|\bugnt|gunta|libertine)", low):
        return True
    return False


def is_repetition(text):
    # split into sentences and look for repeats
    sents = re.split(r'[\.\n\?!]+', text)
    sents = [s.strip() for s in sents if s.strip()]
    if not sents:
        return False
    c = Counter(sents)
    most_common, freq = c.most_common(1)[0]
    if freq >= 4:
        return True
    # detect long repeated substring
    if re.search(r'(.{20,200})\1{2,}', text):
        return True
    return False


def is_system_leak(text):
    low = text.lower()
    # typical system prompts leaking
    if 'you are a helpful assistant' in low or 'check if it contains any false presuppositions' in low:
        return True
    # other prompt-like artifacts
    if re.search(r'you are a .*assistant|before answering', low):
        return True
    return False


def is_short(text):
    ws = re.findall(r"\w+", text)
    return len(ws) <= 6


def is_garbage(text):
    if not text:
        return True
    # fraction of alnum
    alnum = len(re.findall(r'\w', text))
    total = max(1, len(text))
    if alnum / total < 0.45:
        return True
    # excessive punctuation
    if re.search(r'[\^\~\*]{5,}', text):
        return True
    return False


def classify(text):
    if is_garbage(text):
        return 'garbage'
    if is_repetition(text):
        return 'repetition'
    if is_injected(text):
        return 'injected_token'
    if is_system_leak(text):
        return 'system_leak'
    if is_short(text):
        return 'short'
    return 'plausible'


def clean_text(text, maxlen=300):
    if not text:
        return ''
    t = text
    # remove common injected tokens
    for token in BLOCKLIST_TOKENS:
        t = re.sub(re.escape(token), '', t, flags=re.I)
    # collapse repeated blocks
    t = re.sub(r'(.{20,200})\1{2,}', r'\1', t, flags=re.S)
    # remove sequences of the same word repeated many times
    t = re.sub(r'\b(\w+)(?:\s+\1){5,}\b', r'\1', t, flags=re.I)
    # keep first two sentences
    parts = re.split(r'(?<=[\.\?!])\s+', t.strip())
    keep = ' '.join([p.strip() for p in parts[:2] if p.strip()])
    keep = html.unescape(keep)
    if len(keep) > maxlen:
        return keep[:maxlen].rsplit(' ', 1)[0] + '...'
    return keep


def summarize_folder(folder, samples_per_cat=3):
    # support files like base.csv and base_multiturn.csv etc.
    patterns = [
        ('base', 'base*.csv'),
        ('critical', 'critical*.csv'),
        ('presupposition', 'presupposition*.csv'),
    ]
    report = []
    for name, pattern in patterns:
        matches = sorted(glob.glob(os.path.join(folder, pattern)))
        if not matches:
            continue
        # prefer the exact name if present, else take first match
        chosen = None
        for m in matches:
            if os.path.basename(m) == f"{name}.csv":
                chosen = m
                break
        if not chosen:
            chosen = matches[0]

        rows = []
        with open(chosen, newline='', encoding='utf-8') as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                rows.append(r)

        total = len(rows)
        cats = Counter()
        samples = defaultdict(list)
        for r in rows:
            # collect response columns: Response_1, Response, Response_Turn_0...
            resp_keys = [k for k in r.keys() if k and k.lower().startswith('response')]
            # sort keys to keep turn order when present
            def _key_sort(k):
                m = re.search(r"(\d+)$", k)
                return int(m.group(1)) if m else 0
            resp_keys = sorted(resp_keys, key=_key_sort)
            parts = [r.get(k, '') for k in resp_keys]
            text = '\n'.join([p for p in parts if p])
            q = r.get('Question', '')
            cat = classify(text)
            cats[cat] += 1
            if len(samples[cat]) < samples_per_cat:
                samples[cat].append({'q': q, 'raw': text, 'first': first_sentence(text), 'cleaned': clean_text(text)})

        report.append({'name': name, 'total': total, 'counts': cats, 'samples': samples, 'file': os.path.basename(chosen)})
    return report


def print_report(report, model_name=None, folder=None):
    header = f"{model_name or 'Model'} — False Presuppositions Model Output Analysis"
    print(header)
    if folder:
        try:
            rel = os.path.relpath(folder, os.getcwd())
            display_folder = './' + rel if not rel.startswith('..') else folder
        except Exception:
            display_folder = folder
        print(f"Folder: {display_folder}")
    for sec in report:
        print('\n== {} =='.format(sec['name']))
        total = sec['total']
        print(f"Total prompts: {total}")
        # order categories by desired order
        order = ['plausible', 'repetition', 'system_leak', 'injected_token', 'short', 'garbage']
        for cat in order:
            cnt = sec['counts'].get(cat, 0)
            pct = (cnt / total * 100) if total else 0.0
            print(f" - {cat:13}: {cnt:6d} ({pct:5.1f}%)")

        print('\nSample cleaned examples (per category):')
        for cat in order:
            items = sec['samples'].get(cat, [])
            if not items:
                continue
            print(f"  - Category: {cat}")
            for it in items:
                raw_trunc = (it['raw'].replace('\n', ' ')).strip()
                if len(raw_trunc) > 300:
                    raw_trunc = raw_trunc[:300].rsplit(' ', 1)[0] + '...'
                print(f"    Q: {it['q']}")
                print(f"    Raw (truncated): {raw_trunc}")
                print(f"    First sentence: {it['first']}")
                print(f"    Cleaned: {it['cleaned']}\n")


def main():
    p = argparse.ArgumentParser(description='Report generator for false-presuppositions outputs')
    p.add_argument('folder', nargs='?', default=None, help='Path to model output folder')
    p.add_argument('--model-name', default=None, help='Model display name')
    args = p.parse_args()
    folder = args.folder or os.getcwd()
    report = summarize_folder(folder)
    if not report:
        print('No recognized CSV files found in', folder)
        return
    print_report(report, model_name=args.model_name, folder=folder)


if __name__ == '__main__':
    main()
