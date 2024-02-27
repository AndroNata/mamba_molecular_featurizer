from typing import Optional, List, Any
import torch
from torch import nn

import lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig

class SymbolPredictor(pl.LightningModule):

    def __init__(self,
                 d_model=256,
                 n_layer=10,
                 vocab_size=421,
                 top_k=2
                 ):
        super().__init__()
        self.save_hyperparameters()
        mamba_config = MambaConfig(
            d_model=d_model,
            n_layer=n_layer,
            vocab_size=vocab_size
        )
        self.model = MambaLMHeadModel(
            config=mamba_config,
            dtype=torch.float,
            device="cuda"
        )    # Predict a symbol by all it's neighbour symbols

        self.criterion = nn.CrossEntropyLoss(reduction='mean')

        self.val_exact_match_scores = []
        self.val_partial_match_scores = []
        self.val_precision_scores = []
        self.val_loss = []

        self.test_exact_match_scores = []
        self.test_partial_match_scores = []
        self.test_precision_scores = []
        self.test_predicted_labels = []
        self.test_tgt_labels = []
        self.test_loss = []

    def on_train_start(self) -> None:
        print(f"\nBatch hard mining strategy: {self.hparams.batch_hard_mining}")

    def compute_logits_and_labels(self, input_ids):
        lm_logits = self.model(input_ids).logits  # b_sz,max_len,dict_size
        b_sz, max_len, dict_size = lm_logits.size()
        shifted_logits = lm_logits[:, :-1, :].contiguous()  # b_sz,max_len-1,dict_size
        shifted_labels = input_ids[:, 1:].contiguous()  # b_sz,max_len-1
        return shifted_logits.view(-1, dict_size), shifted_labels.view(-1)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        input_ids = batch.pop("input_ids")  # b_sz,max_len
        loss = self.criterion(*self.compute_logits_and_labels(input_ids))
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def check_performance(self, batch):
        input_ids = batch.pop("input_ids")  # b_sz,max_len
        logits, labels = self.compute_logits_and_labels(input_ids)  # (b_sz*(max_len-1),dict_size),  (b_sz*(max_len-1))
        loss = self.criterion(logits, labels)

        pred_labels = torch.topk(logits, self.hparams.top_k, -1).indices  # (b_sz*(max_len-1),top_k)

        pred_labels = list(pred_labels.cpu().numpy())
        tgt_labels = list(labels.cpu().numpy()) # list of b_sz length like [array([0]), array([1])]

        return *calculate_metrics(pred_labels, tgt_labels, max_k=self.hparams.top_k), loss


    def validation_step(self, batch, batch_idx):
        ex, pt, ap, loss = self.check_performance(batch)

        self.val_exact_match_scores += ex
        self.val_partial_match_scores += pt
        self.val_precision_scores += ap
        self.val_loss.append(loss)

    def on_validation_epoch_end(self) -> None:
        self.log("val/exact_match_acc", torch.tensor(self.val_exact_match_scores).float().mean(),
                 on_step=False, on_epoch=True)
        print("val/exact_match_acc:", torch.tensor(self.val_exact_match_scores).float().mean())
        self.log("val/partial_match_acc", torch.tensor(self.val_partial_match_scores).float().mean(),
                 on_step=False, on_epoch=True)
        self.log("val/mAP", torch.tensor(self.val_precision_scores).mean(),
                 on_step=False, on_epoch=True)
        self.log("val/loss", torch.tensor(self.val_loss).mean())

    def on_validation_end(self) -> None:
        self.val_exact_match_scores.clear()
        self.val_partial_match_scores.clear()
        self.val_precision_scores.clear()
        self.val_loss.clear()


    def test_step(self, batch, batch_idx):
        ex, pt, ap, loss = self.check_performance(batch)

        self.test_exact_match_scores += ex
        self.test_partial_match_scores += pt
        self.test_precision_scores += ap
        self.test_loss.append(loss)


    def on_test_epoch_end(self) -> None:
        # TODO Log histograms
        # TODO Log most frequently predicted reagents
        self.log("test/exact_match_acc", torch.tensor(self.test_exact_match_scores).float().mean())
        self.log("test/partial_match_acc", torch.tensor(self.test_partial_match_scores).float().mean())
        self.log("test/mAP", torch.tensor(self.test_precision_scores).mean())
        self.log("test/loss", torch.tensor(self.test_loss).mean())

    def on_test_end(self) -> None:
        self.test_exact_match_scores.clear()
        self.test_partial_match_scores.clear()
        self.test_precision_scores.clear()
        self.test_predicted_labels.clear()
        self.test_tgt_labels.clear()
        self.test_loss.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


def calculate_metrics(pred_labels: List[Any], tgt_labels: List[Any], max_k: int):
    exact_match_scores = []
    partial_match_scores = []
    average_precision_scores = []

    for pred, tgt in zip(pred_labels, tgt_labels):
        tgt_set = [tgt]
        tgt_len = len(tgt_set)
        n_overlap = len(set(pred) & tgt_set)

        # P@K for k from 1 to TOP_K
        precisions = [len(set(pred[:k]) & tgt_set) / k for k in range(1, max_k + 1)]

        # Average precision
        hits = [int(pred[k] in tgt_set) for k in range(max_k)]
        average_precision = sum([p * h for p, h in zip(precisions, hits)]) / tgt_len

        exact_hit = average_precision == 1.0
        partial_hit = n_overlap > 0

        exact_match_scores.append(int(exact_hit))
        partial_match_scores.append(int(partial_hit))
        average_precision_scores.append(average_precision)

    return exact_match_scores, partial_match_scores, average_precision_scores


