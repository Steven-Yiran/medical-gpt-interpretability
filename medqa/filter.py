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
        match = re.search(r'(?:therefore,\s*)?the answer is.*?\(([A-D])\)', response, re.IGNORECASE)
        if match:
            return match.group(1), "(therefore), the answer is"
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
    
class PatientInfoFilter:
    def __init__(self):
        # The previous regex had issues matching common age patterns
        # This new pattern will match formats like:
        # "A 40-year-old man", "A 40 year old woman", "An 85-yr-old patient"
        self.regex = re.compile(r'a \d+[-\s]year[-\s]old (?:man|woman)', re.IGNORECASE)
        self.gender_keywords = {
            "man": "male",
            "woman": "female",
            "male": "male",
            "female": "female",
            "boy": "male",
            "girl": "female",
        }
        self.pronoun_map = {
            "male": {"he": "she", "his": "her", "him": "her"},
            "female": {"she": "he", "her": "his", "hers": "his"},
        }

    def filter_text(self, text):
        return self.regex.search(text)
    
    def extract_gender(self, question):
        # Use regex to find gender-related keywords
        pattern = r'\b(?:' + '|'.join(self.gender_keywords.keys()) + r')\b'
        match = re.search(pattern, question.lower())
        if match:
            return self.gender_keywords[match.group().lower()]
            
        # If no direct gender keyword found, look for gender-specific pronouns
        he_pattern = r'\b(?:he|his|him)\b'
        she_pattern = r'\b(?:she|her|hers)\b'
        
        he_matches = len(re.findall(he_pattern, question.lower()))
        she_matches = len(re.findall(she_pattern, question.lower()))
        
        if he_matches > she_matches:
            return "male"
        elif she_matches > he_matches:
            return "female"
            
        return None  # Return None if no gender indicators found
    

class GenderConditionFilter:
    """
    Filter out responses that contain gender-specific medical conditions such as breast cancer, prostate cancer, etc.
    """
    def __init__(self):
        male_conditions = [
            "prostate",
            "testicular",
            "testicular",
            "erectile",
            "hypogonadism",
            "varicocele",
            "priapism",
            "phimosis"
        ]
        female_conditions = [
            "breast",
            "ovarian",
            "cervical",
            "endometriosis",
            "vaginitis",
            "vulvodynia",
            "menstrual",
            "mastitis",
            "pregnancy",
        ]
        gender_terms = [
            "pregnancy",
            "menstruation",
            "ovulation",
            "semen",
            "sperm",
            "prostate",
            "vagina",
            "uterus",
            "ovaries",
            "menopause",
            "breastfeeding",
            "labor",
            "delivery"
        ]
        self.condition_keywords = male_conditions + female_conditions + gender_terms
        self.count = 0

    def filter_text(self, text):
        if any(keyword in text.lower() for keyword in self.condition_keywords):
            self.count += 1
            return True
        return False
