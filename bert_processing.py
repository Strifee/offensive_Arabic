from data_processing import X,Y
import time
import gc
import pickle
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup

from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split


X_train,X_val,Y_train,Y_val = train_test_split(X,Y,test_size=0.1, random_state = random.seed(42))
tokenizer = BertTokenizer.from_pretrained('aubmindlab/bert-base-arabertv02', do_lower_case=True)
print('Tokenizing data...')

train_inputs, train_masks = preprocessing_for_bert(X_train)
val_inputs, val_masks = preprocessing_for_bert(X_val)



class Classifier :

    def preprocessing_for_bert(data):
        """Perform required preprocessing steps for pretrained BERT.
        @param    data (np.array): Array of texts to be processed.
        @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
        @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                    tokens should be attended to by the model.
        """
        input_ids = []
        attention_masks = []
        for sent in data:
            encoded_sent = tokenizer.encode_plus(
                text=cleaning_content(sent),  
                add_special_tokens=True,        
                max_length=MAX_LEN,             
                pad_to_max_length=True,         
                return_attention_mask=True      
            )
            input_ids.append(encoded_sent.get('input_ids'))
            attention_masks.append(encoded_sent.get('attention_mask'))
        input_ids = torch.tensor(input_ids)
        attention_masks = torch.tensor(attention_masks)

        return input_ids, attention_masks