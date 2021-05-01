
import torch
from transformers import AutoTokenizer, ElectraForSequenceClassification


class_names = ['negative', 'neutral', 'positive']

tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-small-v2-discriminator")
text = "아 더빙.. 진짜 짜증나네요 목소리"
inputs = tokenizer(
  text,
  return_tensors='pt',
  truncation=True,
  max_length=256,
  pad_to_max_length=True,
  add_special_tokens=True
)


# GPU 사용
device = torch.device("cuda")
model = ElectraForSequenceClassification.from_pretrained("monologg/koelectra-small-v2-discriminator").to(device)

model.load_state_dict(torch.load("model.pt"))

model.eval()

input_ids = inputs['input_ids'].to(device)
attention_mask = inputs['attention_mask'].to(device)
output = model(input_ids, attention_mask).logits
_, prediction = torch.max(output, 1)
print(f'Review text: {text}')
print(f'Sentiment  :{prediction}')