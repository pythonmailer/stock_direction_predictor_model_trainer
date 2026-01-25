import torch
import os
from datetime import datetime
import joblib
import numpy as np
from typing import Dict, Tuple
from .utils import EarlyStopper, get_logger
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score
)

class DeepLearningTrainer:
    def __init__(self, model, criterion, optimizer, device, epochs, patience, run_id, model_type="dl_model", threshold=0.4):
        self.logger = get_logger(__name__)

        # For Training
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs
        self.threshold = threshold
        self.patience = patience

        # For Saving
        self.model_type = model_type
        self.run_id = run_id
        self.save_path = "artifacts/trained_models"
        self.temp_checkpoint_path = f"{self.save_path}/{self.model_type}_temp_best.pth"
        
        # Early Stopper
        self.stopper = EarlyStopper(patience=self.patience)

    def train(self, train_loader, val_loader) -> Tuple[torch.nn.Module, Dict]:
        self.logger.info(f"Starting training for {self.model_type}...")
        
        best_prec = 0.0
        final_name = None

        metrics = {"val_prec": 0.0, "final_loss": 1.0, "epochs_run": 0}
        history = []
        
        # Ensure directory exists before starting
        os.makedirs(self.save_path, exist_ok=True)

        for epoch in range(self.epochs):
            # 1. Train & Validate
            train_loss, train_prec, train_rec, train_f1, train_acc, train_auc = self._train_epoch(train_loader)
            val_loss, val_prec, val_rec, val_f1, val_acc, val_auc = self._validate(val_loader)
            
            self.logger.info(f"Epoch {epoch+1}/{self.epochs} | Loss: {val_loss:.4f} | Prec: {val_prec:.2%} | Rec: {val_rec:.2%} | F1: {val_f1:.2%} | Acc: {val_acc:.2%} | AUC: {val_auc:.2%}")

            current_stats = {
                "epoch": epoch+1,
                "train_loss": train_loss,
                "train_prec": train_prec,
                "train_acc": train_acc,
                "train_recall": train_rec,
                "train_f1": train_f1,
                "train_auc": train_auc,
                "val_loss": val_loss,
                "val_prec": val_prec,
                "val_acc": val_acc,
                "val_recall": val_rec,
                "val_f1": val_f1,
                "val_auc": val_auc,
            }

            history.append(current_stats)

            if val_prec > best_prec:
                best_prec = val_prec
                self._save_temp_checkpoint()
                self.logger.info(f"New best found ({best_prec:.2%}) -> Saved to temp file.")

            # 3. Early Stopping
            if self.stopper.early_stop(val_loss):
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                metrics["epochs_run"] = epoch + 1
                break
            
            metrics["final_loss"] = val_loss
            metrics["epochs_run"] = epoch + 1

        metrics["val_prec"] = best_prec
        
        if best_prec > 0:
            self.model.load_state_dict(torch.load(self.temp_checkpoint_path))
            final_name = self._rename_temp_to_final(best_prec)
            metrics["model_path"] = final_name
        else:
            self.logger.warning("No improvement found. No model saved.")

        result = {
            "model_type": self.model_type,
            "metrics": metrics,
            "history": history,
            "model_path": final_name,
            "threshhold": self.threshold,
            "run_id": self.run_id,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        return self.model, result

    def _train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        tp, tn, fp, fn = 0, 0, 0, 0
        all_scores = []
        all_labels = []
        
        for x, y in loader:
            x, y = x.to(self.device).float(), y.to(self.device).float()
            self.optimizer.zero_grad()
            out = self.model(x)
            loss = self.criterion(out, y.view(-1, 1))
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            probs = torch.sigmoid(out)

            all_scores.append(probs.detach().cpu().numpy())
            all_labels.append(y.detach().cpu().numpy())

            preds = (probs > self.threshold).float()
            tp += ((preds == 1) & (y.view(-1, 1) == 1)).sum().item()
            tn += ((preds == 0) & (y.view(-1, 1) == 0)).sum().item()
            fp += ((preds == 1) & (y.view(-1, 1) == 0)).sum().item()
            fn += ((preds == 0) & (y.view(-1, 1) == 1)).sum().item()

        all_scores = np.concatenate(all_scores)
        all_labels = np.concatenate(all_labels)

        try:
            auc = roc_auc_score(all_labels, all_scores)
        except ValueError:
            auc = -1.0
        
        pred_pos = tp + fp
        act_pos = tp + fn
        prec = (tp / (pred_pos)) if (pred_pos) > 0 else 0.0
        rec = (tp / (act_pos)) if (act_pos) > 0 else 0.0

        temp = prec + rec
        f1 = (2 * prec * rec) / (temp) if (temp) > 0 else 0.0

        tot_pred = tp + tn + fp + fn
        acc = (tp + tn) / (tot_pred) if (tot_pred) > 0 else 0.0

        return total_loss / len(loader), prec, rec, f1, acc, auc

    def _validate(self, loader):
        self.model.eval()
        total_loss = 0
        tp, tn, fp, fn = 0, 0, 0, 0
        all_scores = []
        all_labels = []

        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device).float(), y.to(self.device).float()
                out = self.model(x)

                total_loss += self.criterion(out, y.view(-1, 1)).item()
                probs = torch.sigmoid(out)

                all_scores.append(probs.detach().cpu().numpy())
                all_labels.append(y.detach().cpu().numpy())

                preds = (probs > self.threshold).float()
                tp += ((preds == 1) & (y.view(-1, 1) == 1)).sum().item()
                tn += ((preds == 0) & (y.view(-1, 1) == 0)).sum().item()
                fp += ((preds == 1) & (y.view(-1, 1) == 0)).sum().item()
                fn += ((preds == 0) & (y.view(-1, 1) == 1)).sum().item()

        all_scores = np.concatenate(all_scores)
        all_labels = np.concatenate(all_labels)

        try:
            auc = roc_auc_score(all_labels, all_scores)
        except ValueError:
            auc = -1.0

        pred_pos = tp + fp
        act_pos = tp + fn
        prec = (tp / (pred_pos)) if (pred_pos) > 0 else 0.0
        rec = (tp / (act_pos)) if (act_pos) > 0 else 0.0

        temp = prec + rec
        f1 = (2 * prec * rec) / (temp) if (temp) > 0 else 0.0

        tot_pred = tp + tn + fp + fn
        acc = (tp + tn) / (tot_pred) if (tot_pred) > 0 else 0.0

        return total_loss / len(loader), prec, rec, f1, acc, auc

    def _save_temp_checkpoint(self):
        """Overwrites the same file repeatedly"""
        torch.save(self.model.state_dict(), self.temp_checkpoint_path)

    def _rename_temp_to_final(self, score):
        """Renames the temp file to the permanent, descriptive name"""

        final_filename = f"{self.save_path}/{self.model_type}_prec_{score*100:.2f}_{self.run_id}.pth"

        if os.path.exists(final_filename):
            self.logger.warning(f"File {final_filename} already exists. Overwriting...")
            os.remove(final_filename)
            
        os.rename(self.temp_checkpoint_path, final_filename)
        self.logger.info(f"Renamed temp file to: {final_filename}")
        return final_filename


def train_ml_model(model, X_train, y_train, X_val, y_val, run_id, model_type="ml_model"):
    """
    Generic Trainer for Scikit-Learn / XGBoost models
    """
    logger = get_logger(__name__)
    logger.info(f"Training {model_type}...")
    
    # 1. Train
    if "xgb" in model_type.lower():
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model.fit(X_train, y_train)

    # 2. Predict
    preds = model.predict(X_val)

    try:
        probs = model.predict_proba(X_val)[:, 1]
    except AttributeError:
        probs = preds

    # 3. Calculate Metrics
    acc = accuracy_score(y_val, preds)
    prec = precision_score(y_val, preds, zero_division=0)
    rec = recall_score(y_val, preds, zero_division=0)
    f1 = f1_score(y_val, preds, zero_division=0)
    
    try:
        auc = roc_auc_score(y_val, probs)
    except ValueError:
        auc = 0.5

    logger.info(f"Validation Metrics -> Acc: {acc:.2%} | Prec: {prec:.2%} | Rec: {rec:.2%} | F1: {f1:.2%} | AUC: {auc:.2%}")

    # 4. Save
    base_folder = "artifacts/trained_models"
    os.makedirs(base_folder, exist_ok=True)

    filename = f"{base_folder}/{model_type}_prec_{prec:.2f}_{run_id}.joblib"
    joblib.dump(model, filename)
    logger.info(f"Saved model to {filename}")

    # 5. Return Metrics
    result = {
        "run_id": run_id,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": {
            "acc": acc,
            "prec": prec,
            "rec": rec,
            "f1": f1,
            "auc": auc,
        },
        "model_type": model_type,
        "model_path": filename,
    }
    return model, result