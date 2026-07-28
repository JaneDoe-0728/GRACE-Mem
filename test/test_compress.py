import time
from llmlingua import PromptCompressor

compressor = PromptCompressor(
    model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    use_llmlingua2=True,
)

original_prompt = """
User: I'm looking for some new hiking boots. Do you have any recommendations for waterproof boots that are good for multi-day back…rom a 3-day solo camping trip to Big Sur in early April and my current boots did okay, but I think I need something better.
Assistant: Big Sur in April sounds amazing! I'm glad you got to experience it. Now, let's get you into some new boots that'll keep your… recommend trying on several options and taking them for a spin around the block to ensure a comfortable fit. Happy trails!

"""

# 開始計時
start_time = time.time()

results = compressor.compress_prompt_llmlingua2(
    original_prompt,
    rate=0.4,
    force_tokens=['\n', '.', '!', '?', ','],
    chunk_end_tokens=['.', '\n'],
    return_word_label=True,
    drop_consecutive=True
)
end_time = time.time()
elapsed = end_time - start_time

compressed = results['compressed_prompt']
# origin_tokens = results['origin_tokens']
# compressed_tokens = results['compressed_tokens']
compression_rate = results['rate']

print("Compressed prompt:", compressed)
print(f"Compression took {elapsed:.2f} seconds.")
print("Compression rate:", compression_rate)
