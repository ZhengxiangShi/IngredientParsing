import os
import json
import pickle
from tqdm import tqdm
from argparse import ArgumentParser
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import IngredientParser
from preprocess import get_samples, Vocabulary, IngredientDataset


def test(config, output_path):      
    test_samples_ar_gk, _ = get_samples("./data/ar_gk_test.tsv")
    test_samples_ar, _ = get_samples("./data/ar_test.tsv")
    test_samples_gk, _ = get_samples("./data/gk_test.tsv")
    
    with open('./data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    test_dataset_ar_gk = IngredientDataset(test_samples_ar_gk, vocab)
    test_dataset_ar = IngredientDataset(test_samples_ar, vocab)
    test_dataset_gk = IngredientDataset(test_samples_gk, vocab)
    test_dataloader_ar_gk = DataLoader(test_dataset_ar_gk, shuffle=False, batch_size=config["trainer"]["batch_size"])
    test_dataloader_ar = DataLoader(test_dataset_ar, shuffle=False, batch_size=config["trainer"]["batch_size"])
    test_dataloader_gk = DataLoader(test_dataset_gk, shuffle=False, batch_size=config["trainer"]["batch_size"])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using {}'.format(device))
    loss_fn = nn.CrossEntropyLoss(reduction='mean')

    test_model = IngredientParser(vocab.word_embeddings, config["model"]).to(device)
    test_model.load_state_dict(torch.load(os.path.join(output_path, "model.pt")))
    with torch.no_grad():
        test_model.eval()
        test_total_correct = 0
        test_loss = 0
        test_total_tag = 0
        for i, data in enumerate(tqdm(test_dataloader_ar_gk)):
            input_ids, tag_ids, labels = data
            labels = labels.to(device)
            logits = test_model(input_ids.to(device), tag_ids.to(device))
            test_correct_tag = (torch.argmax(logits, dim=-1) == labels).sum()
            test_total_correct += test_correct_tag.item()
            test_total_tag += labels.shape[1]
            loss = loss_fn(logits.squeeze(0), labels.squeeze(0))
            test_loss += loss.item()

        print('AR and GK Test Acc: {}, Test Loss: {}'.format(test_total_correct/test_total_tag,
                                                             test_loss/test_total_tag))

        test_total_correct = 0
        test_loss = 0
        test_total_tag = 0
        for i, data in enumerate(tqdm(test_dataloader_ar)):
            input_ids, tag_ids, labels = data
            labels = labels.to(device)
            logits = test_model(input_ids.to(device), tag_ids.to(device))
            test_correct_tag = (torch.argmax(logits, dim=-1) == labels).sum()
            test_total_correct += test_correct_tag.item()
            test_total_tag += labels.shape[1]
            loss = loss_fn(logits.squeeze(0), labels.squeeze(0))
            test_loss += loss.item()

        print('AR Test Acc: {}, Test Loss: {}'.format(test_total_correct/test_total_tag, test_loss/test_total_tag))

        test_total_correct = 0
        test_loss = 0
        test_total_tag = 0
        for i, data in enumerate(tqdm(test_dataloader_gk)):
            input_ids, tag_ids, labels = data
            labels = labels.to(device)
            logits = test_model(input_ids.to(device), tag_ids.to(device))
            test_correct_tag = (torch.argmax(logits, dim=-1) == labels).sum()
            test_total_correct += test_correct_tag.item()
            test_total_tag += labels.shape[1]
            loss = loss_fn(logits.squeeze(0), labels.squeeze(0))
            test_loss += loss.item()

        print('GK Test Acc: {}, Test Loss: {}'.format(test_total_correct/test_total_tag, test_loss/test_total_tag))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, metavar='PATH', default="./default_path/")
    args = parser.parse_args()

    with open(os.path.join(args.path, 'config.json'), "r") as fp:
        config = json.load(fp)
        
    test(config, args.path)
