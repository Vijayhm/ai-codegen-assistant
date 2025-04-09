from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the fine-tuned model and tokenizer
MODEL_PATH = "./phi2_cli_finetuned"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

def generate_command(user_input):
    input_text = f"User: {user_input}\nAssistant:"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output_ids = model.generate(input_ids, max_length=100)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Test the assistant
query = "how can I see the origin and commit all my changes to Git?"
response = generate_command(query)
print("Generated Command:", response)



# from llama_cpp import Llama

# # Define the model path
# model_path = "/Users/vijaygowda/Desktop/AI software engineer/CodeLlama-7B-GGUF/codellama_7B_Q4.gguf"
# # model_path = "/Users/vijaygowda/Desktop/AI software engineer/CodeLlama-7B-GGUF/codellama_7B_Q8.gguf" 

# # Load the model with optimized settings
# llm = Llama(
#     model_path=model_path,
#     # n_gpu_layers=1,  # Try with 1 GPU layer (Metal support) or 0 for CPU
#     verbose=False,
#     n_ctx=4096  # Increase context window
# ) 

# # Define prompt
# prompt = """
#         You are a professional software engineer and now your job is to write a best optimized code for the following problem. 
        
#         Problem statement: Write single optimized Python function to do this coding problem below i dont need any other text or comments in the code i just need the code.

#         Minimum Window Substring
#         Given two strings s and t, return the shortest substring of s such that every character in t, including duplicates, is present in the substring. If such a substring does not exist, return an empty string "".

#         You may assume that the correct output is always unique.

#         Example 1:

#         Input: s = "OUZODYXAZV", t = "XYZ"

#         Output: "YXAZ"
#         Explanation: "YXAZ" is the shortest substring that includes "X", "Y", and "Z" from string t.

#         Example 2:

#         Input: s = "xyz", t = "xyz"

#         Output: "xyz"
#         Example 3:

#         Input: s = "x", t = "xy"

#         Output: ""
#         Constraints:

#         1 <= s.length <= 1000
#         1 <= t.length <= 1000
#         s and t consist of uppercase and lowercase English letters.

        
#         Note: 
#         1. Do not include any other text or comments. 
#         2. Just answer this one question.
#         3. start and end the code with this key "##code##"
#         """


# # Run inference with increased max tokens
# output = llm(prompt, max_tokens=1000, temperature=1.0)

# # Print output
# print("Generated Code:\n", output["choices"][0]["text"].strip())
