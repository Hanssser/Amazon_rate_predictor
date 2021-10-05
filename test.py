# import argparse
# import os
#
# import torch
# import torch.nn as nn
# import tqdm
# from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
#                              confusion_matrix)
# from torch.utils.data import DataLoader
# from transformers import (BertConfig, BertForSequenceClassification,
#                           BertTokenizerFast,
#                           DistilBertForSequenceClassification,
#                           DistilBertTokenizerFast)
#
# from data import AmazonTestDataset
#
#
# if __name__ == '__main__':
#     os.environ["TOKENIZERS_PARALLELISM"] = "false"
#     torch.manual_seed(42)
#     torch.set_grad_enabled(False)
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-b', '--batch_size', type=int, default=128)
#     parser.add_argument('-m', '--model', type=str, default='distilbert', choices=['bert', 'distilbert'])
#     args = parser.parse_args()
#     print(args)
#
#     ## model
#     if args.model == 'bert':
#         model_name = 'bert-base-uncased'
#         tokenizer = BertTokenizerFast.from_pretrained(model_name)
#         model = BertForSequenceClassification.from_pretrained(model_name, output_attentions=False, output_hidden_states=False, return_dict=False)
#     elif args.model == 'distilbert':
#         model_name = 'distilbert-base-uncased'
#         tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
#         model = DistilBertForSequenceClassification.from_pretrained(model_name, output_attentions=False, output_hidden_states=False, return_dict=False)
#
#     model.classifier = nn.Linear(model.classifier.in_features, 5)
#     model.num_labels = 5
#     model.load_state_dict(torch.load('final_result/distilbert.pth',map_location="cpu"))
#     print(model)
#
#     model.eval()
#     model = model.to(device)
#
#     testset = AmazonTestDataset(tokenizer, model_name, 'dataset/zelda_usa.json')
#     test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=8)
#     print(f'Test set: {len(testset)}')
#
#     all_preds, all_labels = [], []
#
#     for input_ids, attention_masks, labels in tqdm.tqdm(test_loader):
#         input_ids = input_ids.to(device)
#         attention_masks = attention_masks.to(device)
#         labels = labels.to(device)
#
#         loss, logits = model(
#             input_ids,
#             attention_mask=attention_masks,
#             labels=labels)
#
#         preds = logits.argmax(dim=-1).cpu().tolist()
#         all_preds.extend(preds)
#         all_labels.extend(labels.cpu().tolist())
#
#     print(f'Accuracy: {accuracy_score(all_labels, all_preds)}')
#     cmatrix = confusion_matrix(all_labels, all_preds)
#     disp = ConfusionMatrixDisplay(cmatrix, display_labels=[1, 2, 3, 4, 5])
#     disp = disp.plot(cmap='Blues')
#     disp.figure_.savefig('confusion_matrix.png')
