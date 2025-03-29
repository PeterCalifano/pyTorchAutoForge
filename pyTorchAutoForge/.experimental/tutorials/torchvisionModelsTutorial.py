'''
Script implementing code from PyTorch tutorial about torchvision pretrained models and classes, PeterC, 23-07-2024.
Reference: https://pytorch.org/vision/stable/models.html
'''
# NOTE: Before using the pre-trained models, one must preprocess the image (resize with right resolution/interpolation, apply inference transforms,
# rescale the values etc). There is no standard way to do this as it depends on how a given model was trained.

from torchvision.io.image import read_image
from torchvision.utils import draw_bounding_boxes
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.transforms.functional import to_pil_image
import torch, torchvision
from torchvision.models import resnet50, ResNet50_Weights

TORCH_HOME = '/home/peterc/devDir/MachineLearning_PeterCdev/modelZoo'

# NOTE: Torchvision also provides RAFT model for optical flow
#torchvision.models.optical_flow.raft(weights=None)

# To simplify the pre-processing step, torchvision provides the transforms as part of the models
resnet50_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
weights = ResNet50_Weights.IMAGENET1K_V1
preprocessTransform = weights.transforms()
# The transform is simply apply to the input image as: postprocessed = preprocessTransform(inputImage)

# To list all the available models
all_models = torchvision.models.list_models()
classification_mdoels = torchvision.models.list_models(module=torchvision.models)
# To get a model and its weights for instance:
m1 = torchvision.models.get_model("mobilenet_v3_large", weights=None)
weights_m1 = torchvision.models.get_weight("MobileNet_V3_Large_QuantizedWeights.DEFAULT")

# NOTE: torchvision also provides some quantized models, using int8 weights

# %% Example of classification task using pre-trained model
def example_classification():
    from torchvision.io import read_image
    img = read_image("path/to/image.jpg")

    # Step 1: Initialize model with the best available weights
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval()

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()

    # Step 3: Apply inference preprocessing transforms
    batch = preprocess(img).unsqueeze(0)

    # Step 4: Use the model and print the predicted category
    prediction = model(batch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    print(f"{category_name}: {100 * score:.1f}%")


# %% Example of segmentation task using pre-trained model
def example_segmentation():
    from torchvision.io.image import read_image
    img = read_image("path/to/image.jpg")

    # Step 1: Initialize model with the best available weights
    weights = FCN_ResNet50_Weights.DEFAULT
    model = fcn_resnet50(weights=weights)
    model.eval()

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()

    # Step 3: Apply inference preprocessing transforms
    batch = preprocess(img).unsqueeze(0)

    # Step 4: Use the model and visualize the prediction
    prediction = model(batch)["out"]
    normalized_masks = prediction.softmax(dim=1)
    class_to_idx = {cls: idx for (idx, cls) in enumerate(
        weights.meta["categories"])}
    mask = normalized_masks[0, class_to_idx["dog"]]
    to_pil_image(mask).show()

# %% Example of object detection task using pre-trained model
def example_detection():
    
    img = read_image("path/to/image.jpg")

    # Step 1: Initialize model with the best available weights
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
    model.eval()

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()

    # Step 3: Apply inference preprocessing transforms
    batch = [preprocess(img)]

    # Step 4: Use the model and visualize the prediction
    prediction = model(batch)[0]
    labels = [weights.meta["categories"][i] for i in prediction["labels"]]
    box = draw_bounding_boxes(img, boxes=prediction["boxes"],
                            labels=labels,
                            colors="red",
                            width=4, font_size=30)
    im = to_pil_image(box.detach())
    im.show()

if __name__ == "__main__":
    example_classification() # All models are pre-trained on ImageNet dataset
    print("Done with classification!")
    example_segmentation() # All models are pre-trained on COCO val2017 dataset
    print("Done with segmentation!")
    example_detection() 
    print("Done with detection!")