from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from transformers import pipeline, TrainingArguments

model_name = "facebook/opt-350m"
generator = pipeline("text-generation", model=model_name, device="cuda")

input_text = "비트코인 가격이 급등하며 시장의 주목을 받고 있습니다. 전문가들은 어떤 의견을 내놓고 있을까요?"
output = generator(
    input_text,
    max_length=250,
    num_return_sequences=1,
    truncation=True,
    repetition_penalty=1.2,
)

print(output[0]["generated_text"])

dataset = load_dataset(
    "json",
    data_files="corpus.jsonl",
    split="train",
)


def map_to_texts(examples):
    return {"text": examples["prompt"] + " " + examples["completion"]}


dataset = dataset.map(map_to_texts)

train_valid_split = dataset.train_test_split(test_size=0.2)
train_dataset = train_valid_split["train"]
validation_dataset = train_valid_split["test"]

training_args = TrainingArguments(
    output_dir="/tmp",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="steps",
    eval_steps=1,
    save_steps=500,
    logging_dir="./logs",
    logging_steps=1,
)
trainer = SFTTrainer(
    model_name,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    args=training_args,
    dataset_text_field="text",
)
trainer.train()
trainer.evaluate()


model_path = "/tmp/checkpoint-30"
generator = pipeline("text-generation", model=model_path, device="cuda")
output = generator(
    input_text,
    max_length=250,
    num_return_sequences=1,
    truncation=True,
    repetition_penalty=1.2,
)

print(output[0]["generated_text"])
