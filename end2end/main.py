import torch
import os
import json
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset.grid_loader import GRIDDataset
from dataset.digit_loader import DigitDataset
from models.verification_model import LipTextVerificationModel
from training.train import train_one_epoch
from evaluation.evaluate import evaluate_model
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.dataset_utils import Config, collate_fn_factory



def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main():
    writer = None
    try:
        # Detect if spawned via torchrun
        if "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["LOCAL_RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            # Only initialize if NOT ALREADY initialized by torchrun natively
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
            device = torch.device(f"cuda:{local_rank}")
        else:
            # Standalone spawn logic
            setup(0, 1)
            local_rank = 0
            world_size = 1
            device = torch.device("cuda:0")

        config = Config()
        
        if config.dataset == "digit":
            train_dataset = DigitDataset(config.processed_data_dir, split="train", max_frames=config.max_frames, use_audio=config.use_audio)
            val_dataset = DigitDataset(config.processed_data_dir, split="val", max_frames=config.max_frames, use_audio=config.use_audio)
        else:
            train_dataset = GRIDDataset(config.processed_data_dir, split="train", max_frames=config.max_frames, use_audio=config.use_audio)
            val_dataset = GRIDDataset(config.processed_data_dir, split="val", max_frames=config.max_frames, use_audio=config.use_audio)
        
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
        
        transcript_pool = [s.get("transcript", "") for s in getattr(train_dataset, "samples", [])]
        custom_collate = collate_fn_factory(config.dataset, transcript_pool=transcript_pool)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler, collate_fn=custom_collate, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, sampler=val_sampler, collate_fn=custom_collate, num_workers=4, pin_memory=True)
        
        if config.dataset == "digit":
            vocab_size = 13  # 0-9, !, <pad>, <unk>
        else:
            vocab_size = 41  # PhonemeConverter.vocab_size (39+pad+unk)
        model = LipTextVerificationModel(
            vocab_size=vocab_size,
            lip_backbone_pretrained=config.lip_backbone_pretrained
        ).to(device)
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        
        optimizer = Adam(model.parameters(), lr=config.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        
        if local_rank == 0:
            os.makedirs(config.output_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=os.path.join(config.output_dir, 'logs'))
            best_auc = 0.0
            
        for epoch in range(1, config.epochs + 1):
            train_sampler.set_epoch(epoch)
            train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, writer if local_rank == 0 else None)
            
            metrics = evaluate_model(model, val_loader, device)
            scheduler.step(metrics['roc_auc'])
            
            if local_rank == 0:
                print(
                    f"Epoch {epoch}/{config.epochs} | Train Loss: {train_loss:.4f} | "
                    f"Val ROC AUC: {metrics['roc_auc']:.4f} | "
                    f"Val Ver Acc: {metrics['verification_accuracy']:.4f}"
                )
                writer.add_scalar('Logic/roc_auc', metrics["roc_auc"], epoch)
                writer.add_scalar('Accuracy/verify', metrics["verification_accuracy"], epoch)
                
                # Save best model checkpoint
                if metrics['roc_auc'] > best_auc:
                    best_auc = metrics['roc_auc']
                    torch.save(model.module.state_dict(), os.path.join(config.output_dir, "best_model.pt"))
                
                # Save periodic checkpoint
                torch.save(model.module.state_dict(), os.path.join(config.output_dir, f"checkpoint_ep{epoch}.pt"))
    finally:
        if writer is not None:
            writer.close()
        if dist.is_initialized():
            cleanup()

if __name__ == "__main__":
    if "LOCAL_RANK" in os.environ:
        main()
    else:
        world_size = torch.cuda.device_count()
        if world_size > 0:
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "12355"
            import torch.multiprocessing as mp
            # If using mp.spawn manually
            def entry(rank, ws):
                os.environ["LOCAL_RANK"] = str(rank)
                os.environ["WORLD_SIZE"] = str(ws)
                main()
            mp.spawn(entry, args=(world_size,), nprocs=world_size, join=True)
        else:
            print("No CUDA devices available.")
