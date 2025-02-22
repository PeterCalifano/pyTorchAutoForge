wget https://developer.download.nvidia.com/devblogs/speeding-up-unet.7z
7z x speeding-up-unet.7z

wget https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet50v2/resnet50v2.tar.gz
tar xvf resnet50v2.tar.gz # unpack model data into resnet50v2 folder

rm -r speeding-up-unet.7z
rm -r resnet50v2.tar.gz

# Add to gitignore the folders
echo "resnet50v2/" >> .gitignore
echo "speeding-up-unet/" >> .gitignore



