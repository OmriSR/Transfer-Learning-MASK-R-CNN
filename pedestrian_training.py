import os
import pedestrian_dataset as pd
import torch.optim
from engine import train_one_epoch, evaluate
import utils
import torchvision.transforms as T
import torch
import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader

ROOT = os.getcwd()
DATA_PATH = os.path.join(ROOT, 'data', 'dataset')

def build_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    #  replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model



def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main():

    dataset_train = pd.PedestriansDataset(DATA_PATH,
                                          get_transform(train=True),
                                          is_train=True)

    dataset_test = pd.PedestriansDataset(DATA_PATH,
                                         get_transform(train=False),
                                         is_train=False)


    data_loader_train = DataLoader(
        dataset_train, batch_size=4, num_workers=2,
        collate_fn=utils.collate_fn)

    data_loader_test = DataLoader(
        dataset_test, batch_size=4,  num_workers=2,
        collate_fn=utils.collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    # BG + 1 class
    num_classes = 1 + 1
    # uncomment "build_model" to train the model from scratch
    # model = build_model(num_classes)
    # load the finetuned for further finetuning
    model = torch.load(os.path.join(ROOT, 'mask_rcnn_pedestrians2.pt'))

    model.to(device)

    # freeze all layers except the mask rcnn predictor and fast rcnn predictor
    for name, param in model.named_parameters():
        param.requires_grad = "mask_predictor" in name or "box_predictor" in name


    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params,
                                lr=0.000004,
                                momentum=0.7)
    #  learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                    gamma=0.1)

    num_epochs = 5
    print('Training for {} epoch(s)'.format(num_epochs))
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
        lr_scheduler.step()
        evaluate(model, data_loader_test, device=device)

    torch.save(model, 'mask_rcnn_pedestrians2.pt')


if __name__ == '__main__':
    main()
