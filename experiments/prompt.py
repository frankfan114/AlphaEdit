from util.model_utils import ModelAndTokenizer

mt = ModelAndTokenizer(
    model_name="/rds/general/user/ff422/home/FYP/AlphaEdit/edited_model/gpt2xl_cf",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)

tok = mt.tokenizer
model = mt.model

prompt = "Barack Obama was born in"

inputs = tok(prompt, return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_new_tokens=40)

print(tok.decode(out[0], skip_special_tokens=True))
