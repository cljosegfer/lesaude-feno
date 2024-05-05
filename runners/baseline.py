
import torch
import torch.nn as nn
import os
import numpy as np

from tqdm import tqdm

from dataloaders.baseline import CODEsplit
from utils import plot_log, export, get_inputs, find_best_thresholds, metrics_table, json_dump

BATCH_SIZE = 128
NUM_WORKERS = 6

class Runner():
    def __init__(self, device, model, database, model_label = 'baseline'):
        self.device = device
        self.model = model
        self.database = database
        self.model_label = model_label
        if not os.path.exists('output/{}'.format(model_label)):
            os.makedirs('output/{}'.format(model_label))
        
        self.trn_ds = CODEsplit(database, database.trn_idx_dict)
        self.val_ds = CODEsplit(database, database.val_idx_dict)
        self.tst_ds = CODEsplit(database, database.tst_idx_dict)
    
    def train(self, epochs):
        self.model = self.model.to(self.device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4)

        trn_dl = torch.utils.data.DataLoader(self.trn_ds, batch_size = BATCH_SIZE, 
                                             shuffle = True, num_workers = NUM_WORKERS)
        val_dl = torch.utils.data.DataLoader(self.val_ds, batch_size = BATCH_SIZE, 
                                             shuffle = False, num_workers = NUM_WORKERS)

        for epoch in range(epochs):
            # trn_dl, val_dl = self.dataloader.get_train_dataloader(), self.dataloader.get_val_dataloader()
            trn_log = self._train_loop(trn_dl, optimizer, criterion)
            val_log = self._eval_loop(val_dl, criterion)
            plot_log(self.model_label, trn_log, val_log, epoch)
            export(self.model, self.model_label, epoch)
        export(self.model, self.model_label, epoch = None)

    def _train_loop(self, loader, optimizer, criterion):
        log = []
        self.model.train()
        for batch in tqdm(loader):
            # raw, exam_id, label = batch
            raw = batch['X']
            label = batch['y']
            ecg = get_inputs(raw, device = self.device)
            label = label.to(self.device).float()

            logits = self.model.forward(ecg)
            loss = criterion(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            log.append(loss.item())
        return log
    
    def _eval_loop(self, loader, criterion):
        self.model = self.model.to(self.device)
        log = 0
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(loader):
                # raw, exam_id, label = batch
                raw = batch['X']
                label = batch['y']
                ecg = get_inputs(raw, device = self.device)
                label = label.to(self.device).float()

                logits = self.model.forward(ecg)
                loss = criterion(logits, label)

                log += loss.item()
        return log / len(loader)
    
    def eval(self):
        self.model = self.model.to(self.device)
        # val_dl, test_dl = self.dataloader.get_val_dataloader(), self.dataloader.get_test_dataloader()
        val_dl = torch.utils.data.DataLoader(self.val_ds, batch_size = BATCH_SIZE, 
                                             shuffle = False, num_workers = NUM_WORKERS)
        tst_dl = torch.utils.data.DataLoader(self.tst_ds, batch_size = BATCH_SIZE, 
                                             shuffle = False, num_workers = NUM_WORKERS)
        best_f1s, best_thresholds = self._synthesis(val_dl, best_thresholds = None)
        all_binary_results, all_true_labels, metrics_dict = self._synthesis(tst_dl, best_thresholds)
        json_dump(metrics_dict, self.model_label)
    
    def _synthesis(self, loader, best_thresholds = None):
        if best_thresholds == None:
            num_classes = 6
            thresholds = np.arange(0, 1.01, 0.01)  # Array of thresholds from 0 to 1 with step 0.01
            predictions = {thresh: [[] for _ in range(num_classes)] for thresh in thresholds}
            true_labels_dict = [[] for _ in range(num_classes)]
        else:
            all_binary_results = []
            all_true_labels = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(loader):
                # raw, exam_id, label = batch
                raw = batch['X']
                label = batch['y']
                ecg = get_inputs(raw, device = self.device)
                label = label.to(self.device).float()

                logits = self.model(ecg)
                probs = torch.sigmoid(logits)

                if best_thresholds == None:
                    for class_idx in range(num_classes):
                        for thresh in thresholds:
                            predicted_binary = (probs[:, class_idx] >= thresh).float()
                            predictions[thresh][class_idx].extend(
                                predicted_binary.cpu().numpy()
                            )
                        true_labels_dict[class_idx].extend(
                            label[:, class_idx].cpu().numpy()
                        )
                else:
                    binary_result = torch.zeros_like(probs)
                    for i in range(len(best_thresholds)):
                        binary_result[:, i] = (
                            probs[:, i] >= best_thresholds[i]
                        ).float()
                    
                    all_binary_results.append(binary_result)
                    all_true_labels.append(label)
        
        if best_thresholds == None:
            best_f1s, best_thresholds = find_best_thresholds(predictions, true_labels_dict, thresholds)
            return best_f1s, best_thresholds
        else:
            all_binary_results = torch.cat(all_binary_results, dim=0)
            all_true_labels = torch.cat(all_true_labels, dim=0)
            return all_binary_results, all_true_labels, metrics_table(all_binary_results, all_true_labels)