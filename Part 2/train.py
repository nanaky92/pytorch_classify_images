import argparse

import torch

from preprocessing import get_image_datasets, get_dataloaders
from create_model import get_model_arch_from_model_name, get_model
from train_classifier import train_classifier
from checkpoint_handling import store_checkpoint


parser = argparse.ArgumentParser()

parser.add_argument('data_directory')
# ../aipnd-project/flowers/
parser.add_argument(
    '--save_dir',
    default='full_model.pt'
)
parser.add_argument(
    '--arch',
    choices=[
        "vgg11", "vgg13", "vgg16", "vgg19", "vgg11_bn", "vgg13_bn", "vgg16", "vgg19",
    ],
    default="vgg19_bn",
    help="Choose a VGG architecture, with or without batch normalization"
)
parser.add_argument(
    '--learning_rate',
    default=0.001,
    type=float
)
parser.add_argument(
    '--hidden_units',
    nargs=3,
    help='Number of neurons in the two hidden layers in the classifier',
    default=[4096, 2048, 1024],
    type=int
)
parser.add_argument(
    '--epochs',
    default=30,
    type=int
)
parser.add_argument(
    '--gpu',
    action="store_true",
    default=False
)

args = parser.parse_args()
state_dict_checkpoint = 'state_dict_checkpoint.pt'

image_datasets = get_image_datasets(args.data_directory)
dataloaders = get_dataloaders(image_datasets)

model_arch = get_model_arch_from_model_name(args.arch)
model = get_model(model_arch, args.hidden_units)

device = 'cuda' if torch.cuda.is_available() and args.gpu else 'cpu'
print("Running in {}".format(device))

train_classifier(
    model,
    device,
    dataloaders,
    lr=args.learning_rate,
    epochs=args.epochs,
    model_checkpoint=state_dict_checkpoint
)

model.load_state_dict(torch.load(state_dict_checkpoint))
model.class_to_idx = image_datasets['train'].class_to_idx

store_checkpoint(model, model_arch, checkpoint_path=args.save_dir)