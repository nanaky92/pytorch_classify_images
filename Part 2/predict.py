import argparse
import json

import torch

from checkpoint_handling import load_checkpoint
from image_predict import predict_top_k_image

parser = argparse.ArgumentParser()

parser.add_argument('path_to_image')
# ../aipnd-project/flowers/test/1/image_06743.jpg
# ../aipnd-project/flowers/test/2/image_05100.jpg
parser.add_argument('checkpoint')
# 'full_model.pt'
parser.add_argument(
    '--top_k', 
    default=1,
    type=int
)
parser.add_argument('--category_names')
# cat_to_name.json
parser.add_argument(
    '--gpu',
    action="store_true",
    default=False
)

args = parser.parse_args()


def predict(checkpoint_path, image_path, device, top_k, cat_names):
    model, criterion = load_checkpoint(checkpoint_path)

    probs, classes = predict_top_k_image(image_path, model, device, top_k)
    print("Probabilities {}".format(probs))
    print("Classes {}".format(classes))

    if args.category_names:
        with open(args.category_names) as f:
            cat_to_name = json.load(f)
        top_names = [cat_to_name[class_] for class_ in classes]
        print("Classes name {}".format(top_names))


device = 'cuda' if torch.cuda.is_available() and args.gpu else 'cpu'
print("Running in {}".format(device))

predict(args.checkpoint, args.path_to_image, device, args.top_k, args.category_names)
