from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import LoraConfig, get_peft_model

model_name = "facebook/opt-1.3b"


def textGen(model_path):
    generator = pipeline("text-generation", model=model_path, device="cuda")

    input_text = "### Question: 비트코인 가격이 급등하며 시장의 주목을 받고 있습니다. 전문가들은 어떤 의견을 내놓고 있을까요?"
    output = generator(
        input_text,
        max_length=250,
        num_return_sequences=1,
        truncation=True,
        repetition_penalty=1.2,
    )

    print(output[0]["generated_text"])


# test before train
textGen(model_name)

dataset = load_dataset(
    "json",
    data_files="corpus.jsonl",
    split="train",
)

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example["prompt"])):
        text = f"### Question: {example['prompt'][i]}\n### Answer: {example['completion'][i]}"
        output_texts.append(text)
    return output_texts


response_template = "### Answer:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

train_valid_split = dataset.train_test_split(test_size=0.2)
train_dataset = train_valid_split["train"]
validation_dataset = train_valid_split["test"]

lora_config = LoraConfig(
    r=256,
    lora_alpha=64,
    lora_dropout=0.1,
    target_modules=["k_proj", "v_proj", "q_proj", "out_proj"],
)

model = get_peft_model(model, lora_config)

args = SFTConfig(
    output_dir="/tmp",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    eval_strategy="steps",
    eval_steps=20,
    save_total_limit=1,
    logging_dir="./logs",
    logging_steps=1,
    fp16=True,
)
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    args=args,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
)
trainer.train()


model_path = "/tmp/checkpoint-60"

# test after train
textGen(model_path)
