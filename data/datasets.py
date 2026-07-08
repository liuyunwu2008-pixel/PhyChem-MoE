"""Unified MoleculeDataset: loads 9 MoleculeNet benchmarks via DeepChem or fallback CSV.

Supports:
  - Classification (multi-label): BBBP, BACE, ClinTox, SIDER, Tox21
  - Regression: FreeSolv, ESOL, Lipo, QM7
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocessing import LabelStandardizer
from .conformer import generate_conformer, get_atom_types, get_bond_edges


DATASET_CONFIGS = {
    "bbbp":   {"task": "classification", "num_classes": 1,  "type": "binary"},
    "bace":   {"task": "classification", "num_classes": 1,  "type": "binary"},
    "clintox": {"task": "classification", "num_classes": 2,  "type": "multilabel"},
    "sider":  {"task": "classification", "num_classes": 27, "type": "multilabel"},
    "tox21":  {"task": "classification", "num_classes": 12, "type": "multilabel"},
    "freesolv": {"task": "regression", "num_classes": 1, "type": "scalar"},
    "esol":   {"task": "regression", "num_classes": 1, "type": "scalar"},
    "lipo":   {"task": "regression", "num_classes": 1, "type": "scalar"},
    "qm7":    {"task": "regression", "num_classes": 1, "type": "scalar"},
}


def _try_load_deepchem(name: str, data_dir: str) -> Optional[Tuple[List[str], np.ndarray]]:
    """Attempt to load dataset via DeepChem. Returns (smiles_list, labels) or None."""
    try:
        import deepchem as dc
        load_fn = getattr(dc.molnet, f"load_{name}", None)
        if load_fn is None:
            return None
        tasks, datasets, _ = load_fn(featurizer="Raw", split="random", data_dir=data_dir)
        train, valid, test = datasets
        all_data = dc.data.combine_training_test_sets(train, valid, test)

        smiles_list = []
        labels_list = []
        for x, y, w, ids in all_data.itersamples():
            smiles_list.append(x)
            labels_list.append(y)

        labels = np.array(labels_list, dtype=np.float32)
        return smiles_list, labels
    except ImportError:
        return None
    except Exception:
        return None


def _load_from_csv(name: str, data_dir: str) -> Tuple[List[str], np.ndarray]:
    """Simple CSV fallback loader when DeepChem is unavailable."""
    import csv
    csv_path = Path(data_dir) / f"{name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset '{name}' not found at {csv_path}. "
            f"Install deepchem or place the CSV file manually."
        )

    smiles_list = []
    labels_list = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            smiles_list.append(row["smiles"])
            # Assume all other columns are labels
            label_values = [float(v) for k, v in row.items() if k != "smiles"]
            labels_list.append(label_values)

    labels = np.array(labels_list, dtype=np.float32)
    return smiles_list, labels_list


class MoleculeDataset(Dataset):
    """PyTorch Dataset for molecular property prediction across 9 benchmarks.

    Each sample returns:
      - smiles: str
      - labels: dict[task_name -> Tensor]
      - mol_idx: int (for cache lookup)
    """

    def __init__(
        self,
        datasets: List[str],
        data_dir: str = "./data",
        standardizer: Optional[LabelStandardizer] = None,
        fit_standardizer: bool = True,
    ):
        self.datasets = datasets
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.standardizer = standardizer or LabelStandardizer()

        self.samples: List[Dict] = []
        self._load_all_datasets(fit_standardizer)

    def _load_all_datasets(self, fit_standardizer: bool) -> None:
        for name in self.datasets:
            cfg = DATASET_CONFIGS[name]

            result = _try_load_deepchem(name, str(self.data_dir))
            if result is not None:
                smiles_list, labels = result
            else:
                smiles_list, labels = _load_from_csv(name, str(self.data_dir))

            if cfg["task"] == "regression":
                labels = labels.reshape(-1).astype(np.float32)
                nan_mask = ~np.isnan(labels)
                valid_labels = labels[nan_mask]
                if len(valid_labels) == 0:
                    continue
                if fit_standardizer:
                    self.standardizer.fit(valid_labels, name)
                labels_norm = np.full_like(labels, np.nan, dtype=np.float32)
                labels_norm[nan_mask] = self.standardizer.transform(valid_labels, name)
                labels = labels_norm
            else:
                labels = np.nan_to_num(labels, nan=0.0).astype(np.float32)

            for i, smi in enumerate(smiles_list):
                label_val = labels[i] if cfg["task"] == "regression" else labels[i]
                if cfg["task"] == "regression" and np.isnan(label_val):
                    continue
                self.samples.append({
                    "smiles": smi,
                    "labels": {name: torch.tensor(label_val, dtype=torch.float32)},
                    "task_type": {name: cfg["task"]},
                    "mol_idx": len(self.samples),
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]

    def get_standardizer(self) -> LabelStandardizer:
        return self.standardizer


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate a list of samples into a batch."""
    smiles = [s["smiles"] for s in batch]
    mol_indices = torch.tensor([s["mol_idx"] for s in batch], dtype=torch.long)

    # Merge labels across samples
    all_labels: Dict[str, List[torch.Tensor]] = {}
    all_task_types: Dict[str, str] = {}
    for s in batch:
        for task_name, label in s["labels"].items():
            if task_name not in all_labels:
                all_labels[task_name] = []
                all_task_types[task_name] = s["task_type"][task_name]
            all_labels[task_name].append(label)

    merged_labels = {}
    for task_name, label_list in all_labels.items():
        merged_labels[task_name] = torch.stack(label_list, dim=0)

    return {
        "smiles": smiles,
        "mol_indices": mol_indices,
        "labels": merged_labels,
        "task_types": all_task_types,
    }
