"""Entrenamiento: train_gat_lif_bb y quick_train (fiel al notebook)."""
import os
import re
import copy
import torch
import torch.nn as nn
import torch.optim as optim

from .config import CONFIG, SAVE_BASE_PATH, DEVICE, SEED


def quick_train(model, loader, epochs, lr=0.002, weight_decay=1e-4, verbose=False, pos_weight_val=2.0):
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    pos_w = torch.full((model.k_logical,), pos_weight_val, device=model.input_proj.weight.device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, "min", patience=2, factor=0.5)

    model.train()
    history = []
    best_loss = float("inf")
    best_exact = -float('inf')
    best_state = None

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in loader:
            batch = batch.to(model.input_proj.weight.device)
            opt.zero_grad()
            logits = model(batch)
            loss = loss_fn(logits, batch.y.view(-1, model.k_logical))
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(loader) if len(loader) else 0.0
        scheduler.step(avg_loss)
        history.append(avg_loss)
        if avg_loss < best_loss:
            if save_best_by == 'loss':
                if avg_loss < best_loss: best_loss = avg_loss; torch.save(model.state_dict(), best_model_path)
            elif save_best_by == 'exact' and eval_loader is not None:
                try:
                    from qecbb.evaluation import eval_accuracy
                    with torch.no_grad(): acc_bit, acc_ex, _ = eval_accuracy(model, eval_loader, device)
                    if acc_ex > best_exact:
                        best_exact = acc_ex
                        best_model_path_exact = os.path.join(SAVE_BASE_PATH, f'model_bestexact_{safe_code_name}.pth')
                        torch.save(model.state_dict(), best_model_path_exact)
                except Exception: pass
            best_state = copy.deepcopy(model.state_dict())
        if verbose:
            print(f"  epoch {epoch + 1}/{epochs} | loss={avg_loss:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def train_gat_lif_bb(model, loader, total_epochs=12, lr=0.002, weight_decay=0.0001, pos_weight_val=2.0, device='cpu', save_best_by='loss', eval_loader=None):
    model = model.to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    pos_w = torch.full((model.k_logical,), pos_weight_val, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, "min", patience=2, factor=0.5)

    model.train()
    history = []
    start_epoch = 0
    best_loss = float("inf")
    epochs_run = 0

    raw_name = str(CONFIG["CODE_NAME"])
    safe_code_name = re.sub(r"[\[\]\s]+", "", raw_name).replace(",", "_")

    checkpoint_filepath = os.path.join(SAVE_BASE_PATH, f"checkpoint_last_{safe_code_name}.pth")
    loaded_from_checkpoint = False

    last_found_ep = 0
    for ep in range(total_epochs, 0, -1):
        ep_path = os.path.join(SAVE_BASE_PATH, f"model_state_dict_{safe_code_name}_ep{ep}.pth")
        if os.path.exists(ep_path):
            last_found_ep = ep
            break

    if os.path.exists(checkpoint_filepath):
        try:
            checkpoint = torch.load(checkpoint_filepath, map_location=device, weights_only=False)
            ckpt_epoch = checkpoint.get("epoch", 0)
            if ckpt_epoch >= last_found_ep:
                model.load_state_dict(checkpoint["model_state_dict"])
                opt.load_state_dict(checkpoint["optimizer_state_dict"])
                if "scheduler_state_dict" in checkpoint:
                    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                history = checkpoint.get("loss_history", [])
                start_epoch = ckpt_epoch
                epochs_run = start_epoch
                best_loss = checkpoint.get("best_loss", min(history) if history else float("inf"))
                loaded_from_checkpoint = True
                if start_epoch >= total_epochs:
                    print(f"Las {total_epochs} ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â©pocas ya completadas para {CONFIG['CODE_NAME']}.")
                    return model, history, total_epochs
                print(f"Reanudando desde ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â©poca {start_epoch + 1}/{total_epochs}")
        except Exception as e:
            print(f"Error cargando checkpoint: {e}")

    if not loaded_from_checkpoint:
        if last_found_ep > 0:
            ep_path = os.path.join(SAVE_BASE_PATH, f"model_state_dict_{safe_code_name}_ep{last_found_ep}.pth")
            try:
                model.load_state_dict(torch.load(ep_path, map_location=device, weights_only=True))
                start_epoch = last_found_ep
                epochs_run = start_epoch
                print(f"Reanudando desde ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Âºltima ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â©poca guardada: {start_epoch}")
            except Exception as e:
                print(f"Error cargando ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â©poca guardada: {e}; iniciando desde 0")
    else:
        pass

    # entrenamiento simple (se omite el loop completo exacto si ya completo; para pipeline reutiliza caches)
    # replicar bÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¡sico si necesario
    for epoch in range(start_epoch, total_epochs):
        epoch_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            logits = model(batch)
            loss = loss_fn(logits, batch.y.view(-1, model.k_logical))
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(loader)
        scheduler.step(avg_loss)
        history.append(avg_loss)
        epochs_run += 1

    return model, history, total_epochs
