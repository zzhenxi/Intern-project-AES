import logging
import json

logger = logging.getLogger(__name__)

def load_dataset(dataset_name: str, dataset_path: str, n_samples: int = None):
    logger.info(f"Loading dataset from {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f][:n_samples]
        
    logger.info(f"Loaded {len(dataset)} samples from dataset")

    return dataset


# def load_dataset(dataset_name: str, dataset_path: str, n_samples: int = None):
#     logger.info(f"Loading dataset from {dataset_path}")

#     with open(dataset_path, "r", encoding="utf-8") as f:
#         dataset = [json.loads(line) for line in f][:n_samples]
        
#     logger.info(f"Loaded {len(dataset)} samples from dataset")

#     if dataset_name =='asap':
#         logger.info(f"Preprocessing {dataset_name} dataset")
#         splited_dataset = preprocess_asap_dataset(dataset, n_samples)
#         return splited_dataset
#     else:
#         return dataset

# def preprocess_asap_dataset(dataset, n_samples):
#     """Preprocess ASAP dataset by splitting into essay sets"""
#     logger.info("Preprocessing ASAP dataset")
    
#     splited_dataset = {}
#     for sample in dataset:
#         essay_set_id = sample['essay_set']
#         splited_key = f'essay_set_{essay_set_id}'
#         if splited_key not in splited_dataset:
#             splited_dataset[splited_key] = []
#         splited_dataset[splited_key].append(sample)
    
#     essay_set_ids = splited_dataset.keys()
#     for essay_set_id in essay_set_ids:
#         if n_samples and len(splited_dataset[essay_set_id]) > n_samples:
#             splited_dataset[essay_set_id] = splited_dataset[essay_set_id][:n_samples]
    
#     logger.info(f"Split dataset into {len(splited_dataset)} essay sets")
#     return splited_dataset
