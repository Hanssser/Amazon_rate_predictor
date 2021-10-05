import json
import os
import torch
import tqdm
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertTokenizerFast, DistilBertTokenizerFast

class AmazonTrainDataset(Dataset):
    def __init__(self, tokenizer, model_name, path='dataset/Video_Games_5.json'):
        r"""score - integer of [1, 5], labeled as integer of [0, 4]
        """

        self.model_name = model_name
        self.tokenizer = tokenizer
        self.input_ids, self.attention_masks, self.labels = self.load_data(path)

    def load_data(self, path):
        r'''read data from JSON file
        save data as `input_ids.pt`, `attention_masks.pt`, `labels.pt`
        '''
        input_ids_path, attention_masks_path, labels_path = \
            f'dataset/{self.model_name}_input_ids.pt', f'dataset/{self.model_name}_attention_masks.pt', f'dataset/{self.model_name}_labels.pt'
        if os.path.exists(input_ids_path) and os.path.exists(attention_masks_path) and os.path.exists(labels_path):
            self.input_ids, self.attention_masks, self.labels = \
                torch.load(input_ids_path), torch.load(attention_masks_path), torch.load(labels_path)
            return self.input_ids, self.attention_masks, self.labels

        self.input_ids, self.attention_masks, self.labels = [], [], []

        pbar = tqdm.tqdm(total=497577); pbar.set_description('Loading data file')
        f = open(path, 'r')
        for line in f:
            pbar.update()
            data = json.loads(line)
            if 'reviewText' not in data:
                continue

            review = data['reviewText']
            label = int(float(data['overall'])) - 1

            encoded_dict = self.tokenizer(
                review, padding='max_length', truncation=True, max_length=512,
                return_attention_mask=True, return_tensors='pt'
            )

            self.input_ids.append(encoded_dict['input_ids'])
            self.attention_masks.append(encoded_dict['attention_mask'])
            self.labels.append(label)
            
        f.close(); pbar.close()

        self.input_ids = torch.cat(self.input_ids, dim=0)
        self.attention_masks = torch.cat(self.attention_masks, dim=0)
        self.labels = torch.tensor(self.labels)

        # save
        torch.save(self.input_ids, input_ids_path)
        torch.save(self.attention_masks, attention_masks_path)
        torch.save(self.labels, labels_path)

        return self.input_ids, self.attention_masks, self.labels

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_masks[idx], self.labels[idx]
            

class AmazonTestDataset(Dataset):
    def __init__(self, tokenizer, model_name, path='dataset/zelda_usa.json'):
        r'''test dataset'''
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.input_ids, self.attention_masks, self.labels = self.load_data(path)

    def load_data(self, path):
        r"""read data from JSON file"""
        input_ids_path, attention_masks_path, labels_path = \
            f'dataset/t_{self.model_name}_input_ids.pt', f'dataset/t_{self.model_name}_attention_masks.pt', f'dataset/t_{self.model_name}_labels.pt'
        if all([os.path.exists(input_ids_path), os.path.exists(attention_masks_path), os.path.exists(labels_path)]):
            self.input_ids, self.attention_masks, self.labels = \
                torch.load(input_ids_path), torch.load(attention_masks_path), torch.load(labels_path)
            return self.input_ids, self.attention_masks, self.labels
        
        self.input_ids, self.attention_masks, self.labels = [], [] ,[]

        with open(path, 'r') as f:
            data = json.load(f)
        
        for review_dict in tqdm.tqdm(data):
            try:
                review = review_dict['review Text']
                label = int(float(review_dict['review Grade'])) - 1
            except:
                continue

            encoded_dict = self.tokenizer(
                review, padding='max_length', truncation=True, max_length=512,
                return_attention_mask=True, return_tensors='pt'
            )

            self.input_ids.append(encoded_dict['input_ids'])
            self.attention_masks.append(encoded_dict['attention_mask'])
            self.labels.append(label)

        self.input_ids = torch.cat(self.input_ids, dim=0)
        self.attention_masks = torch.cat(self.attention_masks, dim=0)
        self.labels = torch.tensor(self.labels)

        # save
        torch.save(self.input_ids, input_ids_path)
        torch.save(self.attention_masks, attention_masks_path)
        torch.save(self.labels, labels_path)

        return self.input_ids, self.attention_masks, self.labels

    def __len__(self):
        return self.input_ids.shape[0]
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_masks[idx], self.labels[idx]


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.manual_seed(42)
    model_name = 'distilbert-base-uncased'
    
    # tokenizer = BertTokenizerFast.from_pretrained(model_name)
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    dataset = AmazonTrainDataset(tokenizer, model_name, 'dataset/Video_Games_5.json')

    trainsize = int(0.9 * len(dataset)); valsize = len(dataset) - trainsize
    train_set, val_set = random_split(dataset, [trainsize, valsize])
    print(f'Train set: {trainsize}. Val set: {valsize}.')

    train_loader = DataLoader(train_set, batch_size=40, shuffle=False, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=40, shuffle=False, num_workers=8)

    for input_ids, attention_masks, labels in train_loader:
        print(input_ids.shape)
        print(attention_masks.shape)
        print(labels.shape)
        break

    testset = AmazonTestDataset(tokenizer, model_name, 'dataset/zelda_usa.json')
    test_loader = DataLoader(testset, batch_size=40, shuffle=False, num_workers=8)
    print(f'Test set: {len(testset)}')
    
    for input_ids, attention_masks, labels in test_loader:
        print(input_ids.shape)
        print(attention_masks.shape)
        print(labels.shape)
        break


