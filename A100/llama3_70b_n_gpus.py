import transformers
import torch

model_id = "v2ray/Llama-3-70B"

pipeline = transformers.pipeline(
    "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
)
import time
start = time.time()
print(pipeline("Hey how are you doing today?"))
print(time.time()-start)
