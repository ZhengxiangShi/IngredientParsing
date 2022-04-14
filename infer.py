import torch
import os
import json
import pickle
import time
from model import IngredientParser
from argparse import ArgumentParser
from preprocess import convert_number, preprocess, Vocabulary


def parser(ingredient, vocab):
    tag2ids = {'RBS': 0, 'CD': 1, 'SYM': 2, 'DT': 3, '$': 4, "''": 5, 'TO': 6, 'PDT': 7, 'WP': 8, 'RBR': 9, 'POS': 10,
               'VBD': 11, 'PRP$': 12, 'IN': 13, 'VBN': 14, 'VBP': 15, 'FW': 16, 'JJR': 17, 'NNP': 18, 'JJS': 19,
               'VBZ': 20, 'RP': 21, 'WRB': 22, 'LS': 23, 'RB': 24, '.': 25, 'NN': 26, 'PRP': 27, 'JJ': 28, 'VB': 29,
               'CC': 30, 'MD': 31, ',': 32, '``': 33, 'WDT': 34, 'VBG': 35, ':': 36, 'NNS': 37, 'NNPS': 38}
    ingredient = convert_number(ingredient)
    input_ids = torch.tensor(vocab(ingredient))
    sent = preprocess(ingredient)
    assert len(sent) == len(input_ids)
    tag_ids = torch.tensor(list(map(lambda x: tag2ids.get(x[1], 0), sent)))
    return input_ids, tag_ids, sent


if __name__ == '__main__':
    _parser = ArgumentParser()
    _parser.add_argument("--path", type=str, metavar='PATH', default="./default_path/")
    args = _parser.parse_args()
    
    model_path = args.path
    encoder_vocab_path = './data/vocab.pkl'
    with open(encoder_vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    with open(os.path.join(model_path, 'config.json'), "r") as fp:
        config = json.load(fp)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = IngredientParser(vocab.word_embeddings, config["model"]).to(device)
    model.load_state_dict(torch.load(os.path.join(model_path, 'model.pt')))

    label2id = {'DF': 0, 'NAME': 1, 'O': 2, 'QUANTITY': 3, 'SIZE': 4, 'STATE': 5, 'TEMP': 6, 'UNIT': 7}
    id2label = {v: k for k, v in label2id.items()}    
    
    ingredients = [
        "3 cups all-purpose flour",
        "1 teaspoon nutmeg",
        "1 teaspoon ground ginger",
        "2 teaspoons cinnamon",
        "1 teaspoon baking soda",
        "1 teaspoon salt",
        "1 cup chopped pecans",
        "3 ripe bananas",
        "2 cups granulated sugar",
        "1 20-ounce can of diced pineapple, drained",
        "1 cup canola oil",
        "3 large eggs",
        "16 ounces cream cheese",
        "4 ounces unsalted butter",
        "1 teaspoon vanilla extract",
        "½ teaspoon salt",
        "6 cups powdered sugar",
        "½ cup chopped pecans, for decorating",
        "9 tablespoons unsalted butter, at room temperature",
        "1 cup plus 2 tablespoons sugar",
        "3  large eggs",
        "1 ¼ cups all-purpose flour",
        "1 pinch salt",
        "1 cup fresh ricotta",
        "Zest of 1 lemon",
        "1 tablespoon baking powder",
        "1  apple, peeled and grated (should yield about 1 cup)",
    ]   

    print('There are {} samples in total.\n'.format(len(ingredients)))
    
    start = time.time()
    for ingredient in ingredients:
        input_ids, tag_ids, sent = parser(ingredient, vocab)
        with torch.no_grad():
            model.eval()
            logits = model(input_ids.unsqueeze(0).to(device), tag_ids.unsqueeze(0).to(device))
            prediction = torch.argmax(logits, dim=-1)
            predicted_labels = list(map(lambda x: id2label[x], prediction.squeeze(0).tolist()))
            print(ingredient)
            print('Prediction:', [(o[0], p) for o, p in zip(sent, predicted_labels)], '\n')
            # print(ingredient, ": ", predicted_labels, '\n')
    end = time.time()
    print('Infer time per ingredient phrase: {}s'.format((end - start) / len(ingredients)))
