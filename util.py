"""Utils."""


import csv
import os


def parse_csv(file):
    """Helper method for parsing csv files."""
    with open(file, newline='') as f:
        data = list(csv.reader(f))

    return data


def stopword(wstr: str):
    """Helper method for filtering out words with length >= 4."""
    return len(wstr.strip()) < 4 or wstr.startswith('@')


def parse_word(word: str):
    """Helper method for removing symbols and """
    chars = '@#!?;:.,-_ '
    return word.strip(chars)


def parse_sample(file: str):
    """Parse threads to a sample list."""
    words_list = []

    if os.path.isfile(file):
        with open(file, "r") as file:
            for line in file:
                words_list.append([parse_word(w) for w in line.split() if not stopword(w)])
    else:
        raise Exception("File not found!")

    return words_list


def normalize(probs: dict) -> dict:
    """Normalize probability distribution so that the sum would equal to 1."""
    alpha = 1 / sum(probs.values())
    return {k: v * alpha for k, v in probs.items()}
