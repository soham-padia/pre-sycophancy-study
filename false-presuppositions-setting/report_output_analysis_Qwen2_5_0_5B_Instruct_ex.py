#!/usr/bin/env python3
"""Wrapper that runs the generic report against the Qwen2.5-0.5B-Instruct_ex folder.

Run from repository root:
  python false-presuppositions-setting/report_output_analysis_Qwen2_5_0_5B_Instruct_ex.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from report_output_analysis import summarize_folder, print_report


def main():
    folder = os.path.join(os.path.dirname(__file__), 'output', 'Qwen2.5-0.5B-Instruct_ex')
    if not os.path.isdir(folder):
        print('Target folder not found:', folder)
        return
    report = summarize_folder(folder)
    print_report(report, model_name='Qwen2.5-0.5B', folder=folder)


if __name__ == '__main__':
    main()
