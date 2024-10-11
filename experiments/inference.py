import argparse

from transformers import pipeline, set_seed
from transformers import BioGptTokenizer, BioGptForCausalLM


def main():
    parser = argparse.ArgumentParser(description="Generate text using BioGPT")

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    # Load prompt from the user, cannot be empty
    aa("--interactive", action="store_true", help="Whether to run in interactive mode")
    args = parser.parse_args()

    if args.interactive:
        while True:
            inputs = input("Please enter a prompt: ")
            if not inputs:
                print("Prompt cannot be empty")
                continue
            generate_text(inputs)
            

def generate_text(prompt):
    model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
    tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
    # add gpu device
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device="cuda:0")
    set_seed(42)
    results = generator(
        prompt,
        max_length=20,
        truncation=True,
    )

    print("Generated text:")
    for result in results:
        print(result["generated_text"])

if __name__ == "__main__":
    main()