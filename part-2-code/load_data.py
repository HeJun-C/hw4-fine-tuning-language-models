import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import nltk
nltk.download('punkt', quiet=True)
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):
    def __init__(self, data_folder, split):
        '''
        Dataset class for performing data processing for the T5 model.
        Uses 'google-t5/t5-small' tokenizer checkpoint to tokenize both
        encoder and decoder inputs.
        '''
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        
        # Process and tokenize the data
        self.encoder_inputs, self.decoder_inputs, self.decoder_targets = self.process_data(
            data_folder, split, self.tokenizer
        )
        
    def process_data(self, data_folder, split, tokenizer):
        '''
        Load and tokenize the data for the given split.
        '''
        # Load natural language queries and SQL queries
        nl_path = os.path.join(data_folder, f'{split}.nl')
        sql_path = os.path.join(data_folder, f'{split}.sql')
        
        nl_queries = load_lines(nl_path)
        
        # SQL queries only exist for train and dev, not test
        if split != 'test':
            sql_queries = load_lines(sql_path)
        else:
            sql_queries = None
        
        # Tokenize encoder inputs (natural language queries)
        encoder_inputs = tokenizer(
            nl_queries,
            padding=False,  # We'll do dynamic padding in collate_fn
            truncation=True,
            max_length=512,
            return_tensors=None  # Return lists, not tensors
        )['input_ids']
        
        # Tokenize decoder inputs and targets (SQL queries)
        if sql_queries is not None:
            # For decoder input, prepend with a beginning token (using <pad> as BOS)
            # T5 uses pad_token_id as the decoder start token
            decoder_tokenized = tokenizer(
                sql_queries,
                padding=False,
                truncation=True,
                max_length=512,
                return_tensors=None
            )['input_ids']
            
            # Decoder inputs: prepend decoder_start_token_id (which is pad_token_id for T5)
            decoder_inputs = [[tokenizer.pad_token_id] + ids[:-1] for ids in decoder_tokenized]
            # Decoder targets: the actual SQL tokens (shifted by 1)
            decoder_targets = decoder_tokenized
        else:
            decoder_inputs = None
            decoder_targets = None
        
        return encoder_inputs, decoder_inputs, decoder_targets
    
    def __len__(self):
        return len(self.encoder_inputs)
    
    def __getitem__(self, idx):
        encoder_input = torch.tensor(self.encoder_inputs[idx], dtype=torch.long)
        
        if self.decoder_inputs is not None:
            decoder_input = torch.tensor(self.decoder_inputs[idx], dtype=torch.long)
            decoder_target = torch.tensor(self.decoder_targets[idx], dtype=torch.long)
            # Initial decoder input is just the first token (decoder_start_token_id)
            initial_decoder_input = torch.tensor([self.decoder_inputs[idx][0]], dtype=torch.long)
            return encoder_input, decoder_input, decoder_target, initial_decoder_input
        else:
            # For test set, only return encoder input and initial decoder token
            initial_decoder_input = torch.tensor([PAD_IDX], dtype=torch.long)  # pad_token_id as start
            return encoder_input, initial_decoder_input


def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.
    '''
    encoder_inputs = [item[0] for item in batch]
    decoder_inputs = [item[1] for item in batch]
    decoder_targets = [item[2] for item in batch]
    initial_decoder_inputs = [item[3] for item in batch]
    
    # Pad sequences
    encoder_ids = pad_sequence(encoder_inputs, batch_first=True, padding_value=PAD_IDX)
    decoder_input_ids = pad_sequence(decoder_inputs, batch_first=True, padding_value=PAD_IDX)
    decoder_target_ids = pad_sequence(decoder_targets, batch_first=True, padding_value=PAD_IDX)
    initial_decoder_ids = pad_sequence(initial_decoder_inputs, batch_first=True, padding_value=PAD_IDX)
    
    # Create attention masks (1 for real tokens, 0 for padding)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    
    return encoder_ids, encoder_mask, decoder_input_ids, decoder_target_ids, initial_decoder_ids


def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.
    '''
    encoder_inputs = [item[0] for item in batch]
    initial_decoder_inputs = [item[1] for item in batch]
    
    # Pad sequences
    encoder_ids = pad_sequence(encoder_inputs, batch_first=True, padding_value=PAD_IDX)
    initial_decoder_ids = pad_sequence(initial_decoder_inputs, batch_first=True, padding_value=PAD_IDX)
    
    # Create attention mask
    encoder_mask = (encoder_ids != PAD_IDX).long()
    
    return encoder_ids, encoder_mask, initial_decoder_ids


def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn
    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader


def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines


def load_prompting_data(data_folder):
    '''
    Load data for prompting tasks (not used in this assignment but kept for compatibility)
    '''
    train_nl = load_lines(os.path.join(data_folder, 'train.nl'))
    train_sql = load_lines(os.path.join(data_folder, 'train.sql'))
    dev_nl = load_lines(os.path.join(data_folder, 'dev.nl'))
    dev_sql = load_lines(os.path.join(data_folder, 'dev.sql'))
    test_nl = load_lines(os.path.join(data_folder, 'test.nl'))
    
    return train_nl, train_sql, dev_nl, dev_sql, test_nl