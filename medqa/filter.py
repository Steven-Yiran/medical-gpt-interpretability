import json
import re
import sys
import unicodedata

class MultiChoiceFilter:
    # Inspiring from lmeval
    def __init__(self, ignore_case=False, ignore_punctuation=False, regex_pattern=r"[\(\[]([A-D])[\)\]]"):
        
        self.ignore_case = ignore_case
        self.ignore_punctuation = ignore_punctuation
        self.regex_pattern = regex_pattern
        self.regex = re.compile(regex_pattern)
        self.punct_tbl = dict.fromkeys(i for i in range(sys.maxunicode) 
                                       if unicodedata.category(chr(i)).startswith("P"))

    def filter_text(self, text):
        if self.ignore_case:
            text = text.lower()
        if self.ignore_punctuation:
            text = text.translate(self.punct_tbl)
        return text

    def find_match(self, regex, resp, convert_dict={}):
        # look for the regex patter from the end to the start
        # Search from the end to the start by reversing the string
        reversed_resp = resp[::-1]
        match = regex.search(reversed_resp)
        if match:
            # Extract the match and reverse it back to get the correct order
            match = match.group(0)[::-1]
            # Extract the actual letter from the match using the original regex
            match = regex.search(match)
            if match:
                match = match.group(1)
                if isinstance(match, tuple):
                    match = [m for m in match if m][0]
                match = match.strip()
                if match and match in convert_dict:
                    match = convert_dict[match]
        return match


    def extract_answer(self, response, choices=None):
        # give tally of each cases of match
        match = re.search(r'the answer is .(\w).', response)
        if match:
            return match.group(1), "the answer is"
        match = re.search(r'the answer is.*?\[([A-D])\]', response, re.IGNORECASE)
        if match:
            return match.group(1), "the answer is"
        match = re.search(r'answer is .(\w).', response)
        if match:
            return match.group(1), "answer is"
        match = re.search(r'answer is.*?\[([A-D])\]', response, re.IGNORECASE)
        if match:
            return match.group(1), "answer is"
        match = self.find_match(self.regex, response) 
        if match:
            return match, "single letter"
        return "invalid", "invalid"

    def filter_responses(self, responses, choices):
        return [self.extract_answer(resp, choices) for resp in responses]