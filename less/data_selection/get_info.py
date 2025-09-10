"""
    This script is used for getting gradients or representations of a pre-trained model, a lora model, or a peft-initialized model for a given task.
"""

import argparse
import os


os.environ.pop("NCCL_TUNER_CONFIG_PATH", None)
os.environ.pop("NCCL_NET_PLUGIN", None)


import pdb
from copy import deepcopy
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from less.data_selection.collect_grad_reps import (collect_grads, collect_reps,
                                                   get_loss)
from less.data_selection.get_training_dataset import get_training_dataset
from less.data_selection.get_validation_dataset import get_dataset


def load_model(model_name_or_path: str, tokenizer: Any,
               torch_dtype: Any = torch.bfloat16, device: str = "cpu") -> Any:
    """
    Load a model from a given model name or path with proper tokenizer handling.

    Args:
        model_name_or_path (str): The name or path of the model.
        tokenizer (Any): The tokenizer to match vocabulary size with.
        torch_dtype (Any, optional): The torch data type. Defaults to torch.bfloat16.
        device (str, optional): The device to load the model on. Defaults to "cpu".

    Returns:
        Any: The loaded model on the specified device.
    """

    is_peft = os.path.exists(os.path.join(
        model_name_or_path, "adapter_config.json"))
    if is_peft:
        # load this way to make sure that optimizer states match the model structure
        config = LoraConfig.from_pretrained(model_name_or_path)
        print(f"Loading base model from: {config.base_model_name_or_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path, torch_dtype=torch_dtype)
        
        # 关键步骤：在加载PEFT adapter之前调整embedding大小
        print(f"Original vocab size: {base_model.get_input_embeddings().weight.size(0)}")
        print(f"Tokenizer vocab size: {len(tokenizer)}")
        
        if base_model.get_input_embeddings().weight.size(0) != len(tokenizer):
            print("Resizing token embeddings to match tokenizer...")
            base_model.resize_token_embeddings(len(tokenizer))
            print(f"New vocab size: {base_model.get_input_embeddings().weight.size(0)}")
        
        # 现在加载PEFT adapter
        model = PeftModel.from_pretrained(
            base_model, model_name_or_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=torch_dtype)
        
        # 对于非PEFT模型，也需要调整embedding大小
        if model.get_input_embeddings().weight.size(0) != len(tokenizer):
            print("Resizing token embeddings to match tokenizer...")
            model.resize_token_embeddings(len(tokenizer))

    for name, param in model.named_parameters():
        if 'lora' in name or 'Lora' in name:
            param.requires_grad = True
    
    # Move model to specified device
    model = model.to(device)
    return model


parser = argparse.ArgumentParser(
    description='Script for getting validation gradients')
parser.add_argument('--task', type=str, default=None,
                    help='Specify the task from bbh, tydiqa or mmlu. One of variables of task and train_file must be specified')
parser.add_argument("--train_file", type=str,
                    default=None, help="The path to the training data file we'd like to obtain the gradients/representations for. One of variables of task and train_file must be specified")
parser.add_argument(
    "--info_type", choices=["grads", "reps", "loss"], help="The type of information")
parser.add_argument("--model_path", type=str,
                    default=None, help="The path to the model")
parser.add_argument("--max_samples", type=int,
                    default=None, help="The maximum number of samples")
parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                    choices=["float32", "bfloat16"], help="The torch data type")
parser.add_argument("--output_path", type=str,
                    default=None, help="The path to the output")
parser.add_argument("--data_dir", type=str,
                    default=None, help="The path to the data")
parser.add_argument("--gradient_projection_dimension", nargs='+',
                    help="The dimension of the projection, can be a list", type=int, default=[8192])
parser.add_argument("--gradient_type", type=str, default="adam",
                    choices=["adam", "sign", "sgd"], help="The type of gradient")
parser.add_argument("--chat_format", type=str,
                    default="tulu", help="The chat format")
parser.add_argument("--use_chat_format", type=bool,
                    default=True, help="Whether to use chat format")
parser.add_argument("--max_length", type=int, default=2048,
                    help="The maximum length")
parser.add_argument("--zh", default=False, action="store_true",
                    help="Whether we are loading a translated chinese version of tydiqa dev data (Only applicable to tydiqa)")
parser.add_argument("--initialize_lora", default=False, action="store_true",
                    help="Whether to initialize the base model with lora, only works when is_peft is False")
parser.add_argument("--lora_r", type=int, default=8,
                    help="The value of lora_r hyperparameter")
parser.add_argument("--lora_alpha", type=float, default=32,
                    help="The value of lora_alpha hyperparameter")
parser.add_argument("--lora_dropout", type=float, default=0.1,
                    help="The value of lora_dropout hyperparameter")
parser.add_argument("--lora_target_modules", nargs='+', default=[
                    "q_proj", "k_proj", "v_proj", "o_proj"],  help="The list of lora_target_modules")
parser.add_argument("--world_size", type=int, default=1,
                    help="Number of GPUs to use for distributed training")
parser.add_argument("--rank", type=int, default=0,
                    help="Rank of the current process")
parser.add_argument("--local_rank", type=int, default=0,
                    help="Local rank of the current process")
parser.add_argument("--master_addr", type=str, default="localhost",
                    help="Master address for distributed training")
parser.add_argument("--master_port", type=str, default="12355",
                    help="Master port for distributed training")

def setup_distributed(rank, world_size, master_addr, master_port):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()

def main_worker(rank, world_size, args):
    """Main worker function for distributed training"""
    # Setup distributed training
    if world_size > 1:
        setup_distributed(rank, world_size, args.master_addr, args.master_port)
    
    # Set device
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer first (from PEFT adapter path to get correct vocabulary)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # pad token is not added by default for pretrained models
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    # Load model with tokenizer to handle embedding resizing properly
    dtype = torch.float16 if args.torch_dtype == "float16" else torch.bfloat16
    model = load_model(args.model_path, tokenizer, dtype, device)
    
    # Wrap model with DDP if using multiple GPUs
    if world_size > 1:
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    if args.initialize_lora:
        assert not isinstance(model, PeftModel)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
        )
        model = get_peft_model(model, lora_config)

    if isinstance(model, PeftModel):
        model.print_trainable_parameters()

    adam_optimizer_state = None
    if args.info_type == "grads" and args.gradient_type == "adam":
        optimizer_path = os.path.join(args.model_path, "optimizer.pt")
        adam_optimizer_state = torch.load(
            optimizer_path, map_location="cpu")["state"]

    # Prepare dataset
    if args.task is not None:
        dataset = get_dataset(args.task,
                              data_dir=args.data_dir,
                              tokenizer=tokenizer,
                              chat_format=args.chat_format,
                              use_chat_format=args.use_chat_format,
                              max_length=args.max_length,
                              zh=args.zh)
    else:
        assert args.train_file is not None
        dataset = get_training_dataset(
            args.train_file, tokenizer, args.max_length, sample_percentage=1.0)
        columns = deepcopy(dataset.column_names)
        columns.remove("input_ids")
        columns.remove("labels")
        columns.remove("attention_mask")
        dataset = dataset.remove_columns(columns)

    # Create dataloader with distributed sampler if using multiple GPUs
    from transformers import DataCollatorForSeq2Seq
    from torch.utils.data import DataLoader
    
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding="longest")
    
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
        dataloader = DataLoader(
            dataset,
            batch_size=1,  # When getting gradients, we only do this single batch process
            sampler=sampler,
            collate_fn=data_collator
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=1,  # When getting gradients, we only do this single batch process
            collate_fn=data_collator
        )
    
    print(f"Rank {rank}: There are {len(dataset)} examples in the dataset")

    # Adjust output path for distributed training
    if world_size > 1:
        output_path = os.path.join(args.output_path, f"rank_{rank}")
        os.makedirs(output_path, exist_ok=True)
    else:
        output_path = args.output_path

    # Execute the requested operation
    if args.info_type == "reps":
        collect_reps(dataloader, model, output_path,
                     max_samples=args.max_samples)
    elif args.info_type == "grads":
        collect_grads(dataloader,
                      model,
                      output_path,
                      dataset=dataset,
                      proj_dim=args.gradient_projection_dimension,
                      gradient_type=args.gradient_type,
                      adam_optimizer_state=adam_optimizer_state,
                      max_samples=args.max_samples)
    elif args.info_type == "loss":
        get_loss(dataloader, model, output_path)

    # Clean up distributed training
    if world_size > 1:
        cleanup_distributed()

# Main execution
if __name__ == "__main__":
    args = parser.parse_args()
    assert args.task is not None or args.train_file is not None
    
    # Check if running with torchrun (environment variables set by torchrun)
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Running with torchrun - get values from environment
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        # Override args with environment values
        args.rank = rank
        args.world_size = world_size
        args.local_rank = local_rank
        args.master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        args.master_port = os.environ.get("MASTER_PORT", "29500")
        
        # Direct execution with torchrun
        main_worker(rank, world_size, args)
    elif args.world_size > 1:
        # Use multiprocessing for distributed training (fallback)
        mp.spawn(main_worker, args=(args.world_size, args), nprocs=args.world_size, join=True)
    else:
        # Single GPU execution
        main_worker(0, 1, args)
