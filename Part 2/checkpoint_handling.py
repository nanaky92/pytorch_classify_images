import torch
import torch.nn as nn

def store_checkpoint(model, model_arch, checkpoint_path):

    complete_model_dump = {
        "class_name_to_idx": model.class_to_idx,
        "model_state_dict": model.state_dict(),
        "model_classifier": model.classifier,
        "model_conv_layers": model_arch,
        "criterion": nn.NLLLoss
    }

    torch.save(complete_model_dump, checkpoint_path)


def load_checkpoint(checkpoint_path):
    dump = torch.load(checkpoint_path)
    retrieved_model = dump["model_conv_layers"](pretrained=True)

    for param in retrieved_model.parameters():
        param.requires_grad = False

    retrieved_model.classifier = dump["model_classifier"]

    retrieved_model.load_state_dict(dump["model_state_dict"])

    retrieved_model.class_to_idx = dump["class_name_to_idx"]

    criterion = dump["criterion"]()

    return retrieved_model, criterion

