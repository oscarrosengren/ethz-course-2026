import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from hw3.dataset import Normalizer, SO100ChunkDataset, load_zarr, load_and_merge_zarrs
from hw3.model import build_policy

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for states, actions in loader:
        states, actions = states.to(device), actions.to(device)
        optimizer.zero_grad()
        loss = model.compute_loss(states, actions)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)

def main():
    parser = argparse.ArgumentParser(description="Specialized Multicube Trainer")
    parser.add_argument("--zarr", type=Path, required=True, help="Path to processed .zarr")
    parser.add_argument("--policy", default="multitask", help="Policy type (default: multitask)")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--name", default="multicube_v1")
    parser.add_argument("--state-keys", nargs="+", default=None)
    parser.add_argument("--action-keys", nargs="+", default=None)
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load Data ---
    # If no keys provided, use the Exercise 3 defaults we established
    if args.state_keys is None:
        args.state_keys = [
            "original_pos_cube_red", "original_pos_cube_green", "original_pos_cube_blue",
            "state_ee_xyz", "goal_pos", "state_goal", "state_gripper"
        ]
    if args.action_keys is None:
        args.action_keys = ["action_ee_xyz", "action_gripper"]

    states, actions, ep_ends = load_zarr(args.zarr, args.state_keys, args.action_keys)
    print(f"Loaded data: state_dim={states.shape[1]}, action_dim={actions.shape[1]}")

    normalizer = Normalizer.from_data(states, actions)
    dataset = SO100ChunkDataset(states, actions, ep_ends, chunk_size=args.chunk_size, normalizer=normalizer)
    
    n_val = max(1, int(len(dataset) * 0.1))
    train_ds, val_ds = random_split(dataset, [len(dataset)-n_val, n_val])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # --- Model ---
    model = build_policy(
        args.policy, 
        state_dim=states.shape[1], 
        action_dim=actions.shape[1], 
        chunk_size=args.chunk_size
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- Training Loop ---
    ckpt_dir = Path("./checkpoints/multi_cube")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_path = ckpt_dir / f"best_model_{args.name}.pt"
    
    best_val = float('inf')
    for epoch in range(1, args.epochs + 1):
        t_loss = train_one_epoch(model, train_loader, optimizer, device)
        
        # Validation pass
        model.eval()
        v_loss_sum = 0.0
        with torch.no_grad():
            for s, a in val_loader:
                v_loss_sum += model.compute_loss(s.to(device), a.to(device)).item()
        val_loss = v_loss_sum / max(len(val_loader), 1)
        
        scheduler.step()
        
        tag = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "normalizer": {
                    "state_mean": normalizer.state_mean,
                    "state_std": normalizer.state_std,
                    "action_mean": normalizer.action_mean,
                    "action_std": normalizer.action_std,
                },
                "state_keys": args.state_keys,
                "action_keys": args.action_keys,
                "chunk_size": args.chunk_size,
                "policy_type": args.policy,
                "state_dim": int(states.shape[1]),
                "action_dim": int(actions.shape[1]),
            }, save_path)

            tag = " ✓ saved"

        print(f"Epoch {epoch:3d} | train {t_loss:.6f} | val {val_loss:.6f}{tag}")

if __name__ == "__main__":
    main()