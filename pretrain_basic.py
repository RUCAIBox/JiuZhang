from datasets import load_from_disk, Dataset

import json
import math
import logging
import sys
import os

import torch
import transformers
import numpy as np

from dataclasses import dataclass, field
from typing import Optional
from transformers import (
    BertTokenizerFast,
    BartConfig,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    Trainer,
)
from transformers.trainer_utils import is_main_process

from models.modeling_cpt import CPTForPretraining
from models.data_collator import DataCollatorForMTP


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_path: str = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    use_linear_mask: bool = field(
        default=False,
        metadata={
            "help": "Whether or not use linear mask strategy."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_path: str = field(
        default=None,
        metadata={"help": "The input training data file (a text file)."}
    )
    add_token_path: str = field(
        default=None,
        metadata={
            "help": "The additional tokens of your corpus. (a json file that stores a word list)"
            "No need to add tokens if you init model from dapt checkpoint."
        }
    )
    max_input_length: int = field(
        default=256,
        metadata={"help": "The max length for text input ids."}
    )
    mlm_probability: float = field(
        default=0.15,
        metadata={"help": "Ratio of tokens to mask for maksed language modeling loss"}
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    set_seed(training_args.seed)

    config = BartConfig.from_pretrained(model_args.model_path)
    model = CPTForPretraining.from_pretrained(model_args.model_path, config=config)
    tokenizer = BertTokenizerFast.from_pretrained(model_args.model_path, do_lower_case=False)

    if data_args.add_token_path:
        with open(data_args.add_token_path) as f:
            tokenizer.add_tokens(json.load(f))
    model.resize_token_embeddings(len(tokenizer))

    train_dataset = load_from_disk(data_args.data_path)
    
    max_length = min(512, data_args.max_input_length)
    
    def process(examples):
        return tokenizer(examples['content'], examples['analysis'], max_length=max_length, padding=True, truncation=True)
    
    train_dataset = train_dataset.map(process, batched=True)

    data_collator = DataCollatorForMTP(tokenizer=tokenizer, use_linear_mask=model_args.use_linear_mask)

    logger.info("*** Train ***")

    training_args.remove_unused_columns = False

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    train_result = trainer.train()
    trainer.save_model()
    output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
    if trainer.is_world_process_zero():
        with open(output_train_file, "w") as writer:
            logger.info("***** Train results *****")
            for key, value in sorted(train_result.metrics.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")
        trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))


if __name__ == '__main__':
    main()