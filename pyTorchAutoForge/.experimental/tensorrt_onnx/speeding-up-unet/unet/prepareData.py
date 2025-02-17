import torch
import argparse
import numpy as np
from torchvision import transforms                    
from skimage.io import imread
from onnx import numpy_helper
from utils import normalize_volume

def main(args):

    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
      in_channels=3, out_channels=1, init_features=32, pretrained=True)
    model.train(False)
    
    filename = args.input_image
    input_image = imread(filename)
    input_image = normalize_volume(input_image)
    input_image = np.asarray(input_image, dtype='float32')
    
    preprocess = transforms.Compose([
      transforms.ToTensor(),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    
    tensor1 = numpy_helper.from_array(input_batch.numpy())
    with open(args.input_tensor, 'wb') as f:
        f.write(tensor1.SerializeToString())

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model = model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)

    
    tensor = numpy_helper.from_array(output[0].cpu().numpy())
    with open(args.output_tensor, 'wb') as f:
        f.write(tensor.SerializeToString())

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str)
    parser.add_argument('--input_tensor', type=str, default='input_0.pb')
    parser.add_argument('--output_tensor', type=str, default='output_0.pb')
    args=parser.parse_args()
    main(args)
        


