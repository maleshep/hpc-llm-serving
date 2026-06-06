from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
import torch


@dataclass
class SFTConfig:
    model_name: str = "google/gemma-4-31b-it"
    output_dir: str = "./outputs/sft"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    max_seq_length: int = 8192
    bf16: bool = True
    use_4bit: bool = True


def load_model_and_tokenizer(cfg: SFTConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=True,
        load_in_4bit=cfg.use_4bit,
    )
    return model, tokenizer


def setup_lora(model, cfg: SFTConfig):
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    return get_peft_model(model, lora_config)


def main():
    cfg = SFTConfig()
    model, tokenizer = load_model_and_tokenizer(cfg)
    model = setup_lora(model, cfg)

    # Load your SFT dataset (JSONL with 'messages' or 'text' field)
    from datasets import load_dataset
    dataset = load_dataset("json", data_files="./data/sft_train.jsonl", split="train")

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        bf16=cfg.bf16,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        optim="paged_adamw_8bit",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        max_seq_length=cfg.max_seq_length,
        dataset_text_field="messages",
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)


if __name__ == "__main__":
    main()
