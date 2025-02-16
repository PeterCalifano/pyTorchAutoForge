'''
Script implementing the tutorial code for fine tuning a pre-trained object detection model in torchvision.
Reference: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
Created by PeterC, 14-07-2024, with additional features (mflow, optuna) as tests.'''

# Import pre-trained model
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.transforms import v2 as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import torch.utils
import torch.utils.data

from torchvision.transforms.v2 import functional as torchFcn
from torchvision import tv_tensors
from torchvision.ops.boxes import masks_to_boxes
from torchvision.io import read_image

import optuna
import matplotlib
import os
import sys
sys.path.append(os.path.join(
    '/home/peterc/devDir/MachineLearning_PeterCdev/PyTorch/customTorchTools'))

import pyTorchAutoForge
import torch
import torchvision
import mlflow
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

CASE_ID = 1 # 1 or 2

# Download the functions used by the script
os.system("wget --directory-prefix=/home/peterc/devDir/MachineLearning_PeterCdev/PyTorch/tutorials/utilities https://raw.githubusercontent.com/pytorch/vision/main/references/detection/engine.py")
os.system("wget --directory-prefix=/home/peterc/devDir/MachineLearning_PeterCdev/PyTorch/tutorials/utilities https://raw.githubusercontent.com/pytorch/vision/main/references/detection/utils.py")
os.system("wget --directory-prefix=/home/peterc/devDir/MachineLearning_PeterCdev/PyTorch/tutorials/utilities https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_utils.py")
os.system("wget --directory-prefix=/home/peterc/devDir/MachineLearning_PeterCdev/PyTorch/tutorials/utilities https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_eval.py")
os.system("wget --directory-prefix=/home/peterc/devDir/MachineLearning_PeterCdev/PyTorch/tutorials/utilities https://raw.githubusercontent.com/pytorch/vision/main/references/detection/transforms.py")

sys.path.append(os.path.join("/home/peterc/devDir/MachineLearning_PeterCdev/PyTorch/tutorials/utilities"))
import utils
from engine import train_one_epoch, evaluate

# Function to perform transformations on the dataset
def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

# Define the dataset
class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms) -> None:
         self.root = root  # Path to dataset directory
         self.transforms = transforms
         # Load images and sort them
         self.imgs = list(
             sorted(os.listdir(os.path.join(root, "PNGImages"))))
         self.masks = list(
             sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # Load image and mask corresponding to index idx
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = read_image(img_path)
        mask = read_image(mask_path)

        # Apply unique() function to mask
        # This function analyzes the unique values in the mask tensor and returns one ID for each (i.e. one ID per class)
        obj_ids = torch.unique(mask)

        # Remove the first ID, which is the background
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)

        # Split colour encoded masks into a set of binary masks (one per object class)
        masks = mask == obj_ids[:, None, None]

        # Get bounding boxes for each mask
        # torchvision function to get bounding boxes coordinates from masks as [N,4] where N = number of bounding boxes
        boxes = masks_to_boxes(masks)

        # Specify the labels (one class only in this example)
        labels = torch.ones((num_objs,), dtype=torch.int64)

        # SPecify the image ID
        image_id = idx

        # Compute the area of each bounding box
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # Assume all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Wrap image to torchvision tv_tensors.Image
        img = tv_tensors.Image(img)

        # Define target dictionary
        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(
            boxes, format="XYXY", canvas_size=torchFcn.get_size(img))
        
        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        # Apply transform to dataset if specified (NOTE: torch transforms applies to both input and labels)
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.imgs)

# %% Function to get a model to perform instance segmentation combining pre-trained models
def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights="DEFAULT")

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model

# %% MAIN 
def main():
    print('\n\n----------------------------------- TEST SCRIPT: torchvisionFineTuning.py -----------------------------------\n')
    # Dataset details
    # image as torchvision.tv_tensors.Image with shapr [3,H,W]
    # target as a dictionary with keys:
    # 'boxes': [N, 3], where N = number of bounding boxes as [x0, y0, x1, y1] for each box
    # 'labels': torch tensor of shape N (label of each bounding box, with zero being the background class)
    # 'image_id': image identifier
    # 'area': float of shape N, indicating the area of the bounding box
    # 'iscrowd': uint8 torch tensor of shape N, indicating whether the bounding box is a crowd region. If True, the bounding box is ignored.

    import matplotlib.pyplot as plt

    image = read_image(
        "/home/peterc/devDir/MachineLearning_PeterCdev/PyTorch/tutorials/data/PennFudanPed/PNGImages/FudanPed00046.png")
    mask = read_image(
        "/home/peterc/devDir/MachineLearning_PeterCdev/PyTorch/tutorials/data/PennFudanPed/PedMasks/FudanPed00046_mask.png")

    plt.figure(figsize=(16, 8))
    plt.subplot(121)
    plt.title("Image")
    plt.imshow(image.permute(1, 2, 0))
    plt.subplot(122)
    plt.title("Mask")
    plt.imshow(mask.permute(1, 2, 0))
    #plt.show()

    if CASE_ID == 1:
        # Load pre-trained model on COCO
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

        # Replace the classifier with a new one, that has num_classes which is user-defined
        num_classes = 2  # 1 class (person) + background
        # Get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features # This depends on the model before the last classification layer

        # Replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    elif CASE_ID == 2:
        # Load a different backbone model
        backbone = torchvision.models.mobilenet_v2(weights='DEFAULT').features
        backbone.out_channels = 1280 # Define number of output features of the backbone model, required by FasterRCNN to "match" the two models

        # let's make the RPN generate 5 x 3 anchors per spatial location, with 5 different sizes and 3 different aspect ratios. 
        # We have a Tuple[Tuple[int]] because each feature map could potentially have different sizes and aspect ratios
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )  # The RPN is the Region Proposal Network responsible for generating region proposals in the image. 
           # This takes the feature map from the backbone and outputs proposals regions together with their objectness scores.
           # AnchorGenerator specifies the parameters of the anchors (boxes) that the RPN will use.

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=7,
            sampling_ratio=2
        )   # MultiScaleRoIAlign is a layer that performs Region of Interest (RoI) pooling across multiple feature map levels. 
            # It is designed to handle features from different resolutions (scales) and produce fixed-size feature maps for each region proposal. 
            # This is important for object detection tasks where objects can be of varying sizes.

        # Construct the FasterRCNN model using the new backbone, rpn_anchor_generator and box_roi_pool
        model = FasterRCNN(
            backbone,
            num_classes=2,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler
        )

    # Perform example inference to test forward pass
    #model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    #dataset = PennFudanDataset('/home/peterc/devDir/MachineLearning_PeterCdev/PyTorch/tutorials/data/PennFudanPed', get_transform(train=True))
    #data_loader = torch.utils.data.DataLoader(
    #    dataset,
    #    batch_size=2,
    #    shuffle=True,
    #    collate_fn=utils.collate_fn # Don't know what this transform does
    #)

    ## For Training
    #images, targets = next(iter(data_loader))
    #images = list(image for image in images)
    #targets = [{k: v for k, v in t.items()} for t in targets]
    #output = model(images, targets)  # Returns losses and detections
    #print(output)
    ## For inference
    #model.eval()
    #x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    #predictions = model(x)  # Returns predictions
    #print(predictions[0])

    # Training test
    device = pyTorchAutoForge.GetDevice()
    dataset = PennFudanDataset(
        '/home/peterc/devDir/MachineLearning_PeterCdev/PyTorch/tutorials/data/PennFudanPed', get_transform(train=True))
    dataset_test = PennFudanDataset(
        '/home/peterc/devDir/MachineLearning_PeterCdev/PyTorch/tutorials/data/PennFudanPed', get_transform(train=False))
    
    # Split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])


    # Define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=utils.collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        collate_fn=utils.collate_fn
    )

    # Get the model and move it to GPU
    model = get_model_instance_segmentation(num_classes).to(device)

    # Define the optimizer and the learning rate scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(    
        params,
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )


    # Training loop
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader,
                        device, epoch, print_freq=10)
        
        # Update the learning rate
        lr_scheduler.step()

        # Evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    # Visualize output using torchvision functions
    image = read_image(
        '/home/peterc/devDir/MachineLearning_PeterCdev/PyTorch/tutorials/data/PennFudanPed/PNGImages/FudanPed00046.png')
    eval_transform = get_transform(train=False)

    model.eval()
    with torch.no_grad():
        x = eval_transform(image)
        # Convert RGBA -> RGB and move to device
        x = x[:3, ...].to(device)
        # Perform forward pass and get predictions
        predictions = model([x, ])
        pred = predictions[0]

    # Plot image with bounding boxes and masks
    image = (255.0 * (image - image.min()) /
             (image.max() - image.min())).to(torch.uint8)
    image = image[:3, ...]

    pred_labels = [f"pedestrian: {score:.3f}" for label,
                   score in zip(pred["labels"], pred["scores"])]
    pred_boxes = pred["boxes"].long()

    output_image = draw_bounding_boxes(
        image, pred_boxes, pred_labels, colors="red")
    
    masks = (pred["masks"] > 0.7).squeeze(1)
    output_image = draw_segmentation_masks(
        output_image, masks, alpha=0.5, colors="blue")
    
    plt.figure(figsize=(12, 12))
    plt.imshow(output_image.permute(1, 2, 0))
    plt.show()

if __name__ == '__main__':
    main()
