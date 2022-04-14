import os
import nltk
import json
import random
import pickle
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from prettytable import PrettyTable
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import IngredientParser
from preprocess import get_samples, get_augmented_samples, Vocabulary, IngredientDataset

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def compute_entity_level_f1_score(ground_truths, predictions):
    label2id = {'DF': 0, 'NAME': 1, 'O': 2, 'QUANTITY': 3, 'SIZE': 4, 'STATE': 5, 'TEMP': 6, 'UNIT': 7}
    id2label = {v: k for k, v in label2id.items()}
    statistics = {'tp': 0, 'fn': 0, 'fp': 0}
    counting = {k: statistics.copy() for k, v in id2label.items()}
    for g, p in zip(ground_truths, predictions):
        if g != p:
            counting[g]['fn'] += 1
            counting[p]['fp'] += 1
        else:
            counting[g]['tp'] += 1
    
    recall_precision_f1_score = {}    
    for k, v in counting.items():
        recall = v['tp'] / (v['tp'] + v['fn']) if v['tp'] + v['fn'] != 0 else 0
        precision = v['tp'] / (v['tp'] + v['fp']) if v['tp'] + v['fp'] != 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
        recall_precision_f1_score[k] = {'Recall': recall, 'Precision': precision, 'F1_score': f1_score}
    return recall_precision_f1_score
    
def train(config, args):
    if not os.path.exists(args.path):
        os.mkdir(args.path)
        
    # Load dataset
    if config["trainer"]["data_for_train"] == "both":
        train_files = "./data/ar_gk_train.tsv"
    elif config["trainer"]["data_for_train"] == "ar": 
        train_files = "./data/ar_train.tsv"
    elif config["trainer"]["data_for_train"] == "gk": 
        train_files = "./data/gk_train.tsv"
    else:
        print('Please select one of "ar", "gk" or "both" as training datasets.')   
        
    train_and_valid_samples, _ = get_samples(train_files)
    test_samples_ar_gk, _ = get_samples("./data/ar_gk_test.tsv")
    test_samples_ar, _ = get_samples("./data/ar_test.tsv")
    test_samples_gk, _ = get_samples("./data/gk_test.tsv")
    
    random.shuffle(train_and_valid_samples)
    train_size = int(len(train_and_valid_samples)*0.95)
    train_samples, valid_samples = train_and_valid_samples[:train_size], train_and_valid_samples[train_size:]
    if args.aug_size > 0:
        print('Building Augmented Train Set.')
        aug_train_samples = get_augmented_samples(train_files, size=args.aug_size)
        train_samples.extend(aug_train_samples)
    print('Train samples: {}'.format(len(train_samples)))
    print('Valid samples: {}'.format(len(valid_samples)))
    print('Test samples ar: {}'.format(len(test_samples_ar)))
    print('Test samples gk: {}'.format(len(test_samples_gk)))
    print('Test samples ar_gk: {}'.format(len(test_samples_ar_gk)))
    
    if not os.path.exists('./data/vocab.pkl'):
        # Set up the Vocabulary
        print('Constructing Vocabulary.')
        texts = []
        for sample in train_and_valid_samples:
            texts.append(sample['text'])
        for sample in test_samples_ar_gk:
            texts.append(sample['text'])
        vocab = Vocabulary(data=texts)
        with open(os.path.join('.', './data/vocab.pkl'), 'wb') as f:
            pickle.dump(vocab, f)
            print("Saved the vocabulary to '%s'" %os.path.realpath(f.name))
    else:
        print('Using Existing Vocabulary.')
        with open('./data/vocab.pkl', 'rb') as f:
            vocab = pickle.load(f)

    train_dataset = IngredientDataset(train_samples, vocab)
    valid_dataset = IngredientDataset(valid_samples, vocab)
    test_dataset_ar_gk = IngredientDataset(test_samples_ar_gk, vocab)
    test_dataset_ar = IngredientDataset(test_samples_ar, vocab)
    test_dataset_gk = IngredientDataset(test_samples_gk, vocab)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config["trainer"]["batch_size"])
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=config["trainer"]["batch_size"])
    test_dataloader_ar_gk = DataLoader(test_dataset_ar_gk, shuffle=False, batch_size=config["trainer"]["batch_size"])
    test_dataloader_ar = DataLoader(test_dataset_ar, shuffle=False, batch_size=config["trainer"]["batch_size"])
    test_dataloader_gk = DataLoader(test_dataset_gk, shuffle=False, batch_size=config["trainer"]["batch_size"])

    writer = SummaryWriter(log_dir=args.path)
    f_writer = open(os.path.join(args.path, 'output.txt'), 'w')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using {}'.format(device))
    model = IngredientParser(vocab.word_embeddings, config["model"]).to(device)
    count_parameters(model)
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=config["optimizer"]["lr"], betas=(config["optimizer"]["beta1"], config["optimizer"]["beta2"]))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50, 75], gamma=0.2)
    
    with open(os.path.join(args.path, 'config.json'), 'w') as f:
        json.dump(config, f)
        
    for epoch in range(config["trainer"]["epochs"]):
        model.train()
        total_correct = 0
        train_loss = 0
        total_tag = 0
        ground_truth_list = []
        prediction_list = []
        for i, data in enumerate(tqdm(train_dataloader)):
            # if i == 1000: break
            input_ids, tag_ids, labels = data
            labels = labels.to(device) # (batch_size, length)
            logits = model(input_ids.to(device), tag_ids.to(device)) # (batch_size, length, 8)
            predictions = torch.argmax(logits, dim=-1)
            correct_tag = (predictions == labels).sum()
            total_correct += correct_tag.item()
            total_tag += labels.shape[1]
            ground_truth_list.extend(predictions.squeeze(0).tolist())
            prediction_list.extend(labels.squeeze(0).tolist())

            loss = loss_fn(logits.squeeze(0), labels.squeeze(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        f1_score = compute_entity_level_f1_score(ground_truth_list, prediction_list)
        print('Train Acc: {}, Train Loss: {}'.format(total_correct/total_tag, train_loss/total_tag))
        f_writer.write('Epoch {}, Train Acc: {}, Train Loss: {}\n'.format(epoch, total_correct/total_tag, train_loss/total_tag))
        f_writer.write('Epoch {}, F1 Score: {}\n'.format(epoch, f1_score))

        with torch.no_grad():
            model.eval()
            valid_total_correct = 0
            valid_loss = 0
            valid_total_tag = 0
            valid_ground_truth_list = []
            valid_prediction_list = []
            max_acc = 0
            for i, data in enumerate(tqdm(valid_dataloader)):
                input_ids, tag_ids, labels = data
                labels = labels.to(device)
                logits = model(input_ids.to(device), tag_ids.to(device))
                valid_predictions = torch.argmax(logits, dim=-1)
                valid_correct_tag = (valid_predictions == labels).sum()
                valid_total_correct += valid_correct_tag.item()
                valid_total_tag += labels.shape[1]
                loss = loss_fn(logits.squeeze(0), labels.squeeze(0))
                valid_loss += loss.item()
                valid_ground_truth_list.extend(valid_predictions.squeeze(0).tolist())
                valid_prediction_list.extend(labels.squeeze(0).tolist())
            
            valid_f1_score = compute_entity_level_f1_score(valid_ground_truth_list, valid_prediction_list)
            if valid_total_correct/valid_total_tag>max_acc:
                print(f"saved model...")
                torch.save(model.state_dict(), os.path.join(args.path, "model.pt"))
                max_acc = valid_total_correct/valid_total_tag
            
            writer.add_scalars("accuracy", {"train": total_correct/total_tag, "valid": valid_total_correct/valid_total_tag}, epoch)
            writer.add_scalars("loss", {"train": train_loss/total_tag, "validation": valid_loss/valid_total_tag}, epoch)

            print('Valid Acc: {}, Valid Loss: {}\n'.format(valid_total_correct/valid_total_tag, valid_loss/valid_total_tag))
            for k, v in valid_f1_score.items():
                print(k, v)
            f_writer.write('Epoch {}, Valid Acc: {}, Valid Loss: {}\n'.format(epoch, valid_total_correct/valid_total_tag, valid_loss/valid_total_tag))
            f_writer.write('Epoch {}, Valid F1 Score: {}\n'.format(epoch, valid_f1_score))
            
        scheduler.step()
        print('Current_learning_rate:', get_lr(optimizer))
        f_writer.write('Current_learning_rate: {}'.format(get_lr(optimizer)))

    test_model = IngredientParser(vocab.word_embeddings, config["model"]).to(device)
    test_model.load_state_dict(torch.load(os.path.join(args.path, "model.pt")))
    with torch.no_grad():
        test_model.eval()
        test_total_correct = 0
        test_loss = 0
        test_total_tag = 0
        test_ground_truth_list = []
        test_prediction_list = []
        for i, data in enumerate(tqdm(test_dataloader_ar_gk)):
            input_ids, tag_ids, labels = data
            labels = labels.to(device)
            logits = test_model(input_ids.to(device), tag_ids.to(device))
            test_predictions = torch.argmax(logits, dim=-1)
            test_correct_tag = (test_predictions == labels).sum()
            test_total_correct += test_correct_tag.item()
            test_total_tag += labels.shape[1]
            loss = loss_fn(logits.squeeze(0), labels.squeeze(0))
            test_loss += loss.item()
            test_ground_truth_list.extend(test_predictions.squeeze(0).tolist())
            test_prediction_list.extend(labels.squeeze(0).tolist())
        
        test_f1_score = compute_entity_level_f1_score(test_ground_truth_list, test_prediction_list)
        print('AR and GK Test Acc: {}, Test Loss: {}\n'.format(test_total_correct/test_total_tag, test_loss/test_total_tag))
        for k, v in test_f1_score.items():
            print(k, v)
        f_writer.write('\nAR and GK Test Acc: {}, Test Loss: {}\n'.format(test_total_correct/test_total_tag, test_loss/test_total_tag))
        f_writer.write('Test F1 Score: {}\n'.format(test_f1_score))
        
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
        f_writer.write('AR Test Acc: {}, Test Loss: {}\n'.format(test_total_correct/test_total_tag, test_loss/test_total_tag))

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
        f_writer.write('GK Test Acc: {}, Test Loss: {}\n'.format(test_total_correct/test_total_tag, test_loss/test_total_tag))
        
    writer.close()
    f_writer.close()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--aug_size", type=int, default=0)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument("--data_for_train", type=str, default="both")
    parser.add_argument("--config-file", type=str, metavar='PATH', default="config.json")
    parser.add_argument("--path", type=str, metavar='PATH', default="./default_path/")
    args = parser.parse_args()

    nltk.download('averaged_perceptron_tagger')
    seed_torch(args.seed)
    print('Use seed: {}'.format(args.seed))

    with open(args.config_file, "r") as fp:
        config = json.load(fp)
        
    config['optimizer']['lr'] = args.lr
    config["trainer"]["data_for_train"] = args.data_for_train
    config["trainer"]["aug_size"] = args.aug_size
    train(config, args)