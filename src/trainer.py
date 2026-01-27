import torch
import os
from datetime import datetime
import joblib
import numpy as np
import mlflow
from tqdm import tqdm
from typing import Dict, Tuple
from sklearn.ensemble import RandomForestClassifier
import mlflow.xgboost
import xgboost as xgb
from .utils import EarlyStopper, get_logger
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score
)

class DeepLearningTrainer:
    def __init__(self, model, criterion, optimizer, device, epochs, patience, input_example, model_type="dl_model", threshold=0.4):
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
        self.save_path = "artifacts/trained_models"
        self.temp_checkpoint_path = f"{self.save_path}/temp_best.pth"
        self.input_example = input_example
        
        # Early Stopper
        self.stopper = EarlyStopper(patience=self.patience)

    def train(self, train_loader, val_loader) -> Tuple[torch.nn.Module, Dict]:
        self.logger.info(f"Starting training for {self.model_type}...")
        
        best_prec = 0.0
        final_name = None

        metrics = {"best_val_prec": -1.0, "loss": 1.0, "at_epoch": 0}
        
        # Ensure directory exists before starting
        os.makedirs(self.save_path, exist_ok=True)

        for epoch in range(self.epochs):
            # 1. Train & Validate
            train_loss, train_prec, train_rec, train_f1, train_acc, train_auc = self._train_epoch(train_loader)
            val_loss, val_prec, val_rec, val_f1, val_acc, val_auc = self._validate(val_loader)
            
            self.logger.info(f"Epoch {epoch+1}/{self.epochs} | Loss: {val_loss:.4f} | Prec: {val_prec:.2%} | Rec: {val_rec:.2%} | F1: {val_f1:.2%} | Acc: {val_acc:.2%} | AUC: {val_auc:.2%}")

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_prec": train_prec,
                "train_rec": train_rec,
                "train_f1": train_f1,
                "train_acc": train_acc,
                "train_auc": train_auc,
                "val_loss": val_loss,
                "val_prec": val_prec,
                "val_rec": val_rec,
                "val_f1": val_f1,
                "val_acc": val_acc,
                "val_auc": val_auc
            }, step=epoch)

            if val_prec > best_prec:
                best_prec = val_prec
                metrics["loss"] = val_loss
                metrics["at_epoch"] = epoch + 1
                self._save_temp_checkpoint()
                self.logger.info(f"New best found ({best_prec:.2%}) -> Saved!")

            # 3. Early Stopping
            if self.stopper.early_stop(val_loss):
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break

        metrics["best_val_prec"] = best_prec

        result = {
            "metrics": metrics,
            "threshhold": self.threshold,
        }

        self.logger.info(f"Loading best weights from {self.temp_checkpoint_path} for upload...")
        self.model.load_state_dict(torch.load(self.temp_checkpoint_path))

        mlflow.pytorch.log_model(
            pytorch_model=self.model,
            artifact_path="model",

            registered_model_name=self.model_type, 
            
            input_example=self.input_example,
            code_paths=["src/models.py"] 
        )
        self.logger.info("Best model uploaded to DagsHub!")

        return self.model, result

    def _train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        tp, tn, fp, fn = 0, 0, 0, 0
        all_scores = []
        all_labels = []

        pbar = tqdm(loader, desc="Training", leave=False)
        
        for x, y in pbar:
            x, y = x.to(self.device).float(), y.to(self.device).float()
            self.optimizer.zero_grad()
            out = self.model(x)
            loss = self.criterion(out, y.view(-1, 1))
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

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

        pbar = tqdm(loader, desc="Validating", leave=False)

        with torch.no_grad():
            for x, y in pbar:
                x, y = x.to(self.device).float(), y.to(self.device).float()
                out = self.model(x)

                loss_val = self.criterion(out, y.view(-1, 1)).item()
                total_loss += loss_val
                pbar.set_postfix({"loss": f"{loss_val:.4f}"})

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

    # def _rename_temp_to_final(self, score):
    #     """Renames the temp file to the permanent, descriptive name"""

    #     final_filename = f"{self.save_path}/{self.model_type}_prec_{score*100:.2f}_{self.run_id}.pth"

    #     if os.path.exists(final_filename):
    #         self.logger.warning(f"File {final_filename} already exists. Overwriting...")
    #         os.remove(final_filename)
            
    #     os.rename(self.temp_checkpoint_path, final_filename)
    #     self.logger.info(f"Renamed temp file to: {final_filename}")
    #     return final_filename


def train_ml_model(model, X_train, y_train, X_val, y_val, model_type="ml_model"):
    """
    Generic Trainer for Scikit-Learn / XGBoost models
    """
    logger = get_logger(__name__)
    logger.info(f"Training {model_type}...")
    
    # 1. Train
    if "xgb" in model_type.lower():
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            registered_model_name=model_type,
            input_example=X_train[:1]
        )
    else:
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_type,
            input_example=X_train[:1]
        )

    logger.info(f"Saved model to DagsHub.")

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

    mlflow.log_metrics({
        "val_acc": acc,
        "val_prec": prec,
        "val_rec": rec,
        "val_f1": f1,
        "val_auc": auc
    })

    # 5. Return Metrics
    result = {
        "metrics": {
            "acc": acc,
            "prec": prec,
            "rec": rec,
            "f1": f1,
            "auc": auc,
        }
    }
    return model, result