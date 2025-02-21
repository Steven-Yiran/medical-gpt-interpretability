from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re

device = "cuda"  # cuda or cpu
checkpoint = "dmis-lab/meerkat-7b-v1.0"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(
    checkpoint,
    torch_dtype=torch.bfloat16,  # You can choose to use this when there's not enough GPU memory available.
)

# Multi-turn dialogue example
# messages = [
#     {"role": "system", "content": "You are a helpful doctor or healthcare professional. Guide the conversation to provide useful, complete, and scientifically-grounded answers to user questions. You have the option to compose a concise, single-turn conversation if the user's input is comprehensive to provide accurate answers. However, if essential details are missing, you should engage in a multi-turn dialogue, asking follow-up questions to gather a thorough medical history and records.\n\n"},
#     {"role": "user", "content": "Hello, doctor. I'm really concerned about my 10-year-old son. We recently discovered a painless mass in his left testicle, so we brought him to the pediatrician."},
#     {"role": "assistant", "content": "I understand your concern. Let's gather some more information. Has your son experienced any other symptoms along with the mass?"},
#     {"role": "user", "content": "Other than the mass, my son hasn't shown any symptoms. He's been his usual self, playing and eating normally."}
# ]

question = "A 67-year-old man with transitional cell carcinoma of the bladder comes to the physician because of a 2-day history of ringing sensation in his ear. He received this first course of neoadjuvant chemotherapy 1 week ago. Pure tone audiometry shows a sensorineural hearing loss of 45 dB. The expected beneficial effect of the drug that caused this patient's symptoms is most likely due to which of the following actions?"
options = "(A) telomeres. (B) centromeres. (C) histones. (D) ends of the long arms."

messages = [
    {"role": "system", "content": "Answer the multiple-choice question about medical knowledge.\n\n"},
    {"role": "user", "content": question + options},
]

encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

model_inputs = encodeds.to(device)
model.to(device)

generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded)

# search for the answer by searching the formated text "the answer is (A)"
answer = re.search(r"the answer is \([A-D]\)", decoded[0])

if answer:
    print("Answer found:", answer.group())
else:
    print("Answer not found.")
