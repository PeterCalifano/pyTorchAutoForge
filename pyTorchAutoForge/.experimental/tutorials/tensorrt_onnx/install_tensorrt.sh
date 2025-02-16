if conda info --envs | grep -q "tensorrt"; then
    echo "Conda environment tensorrt found. Activating it..."
    conda init bash
    # Activate conda environment
    conda activate tensorrt
else
    echo "Conda environment tensorrt does not exist. Creating it..."
    conda create -n tensorrt python=3.11
    conda activate tensorrt
fi

# Get tar file
#wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.8.0/tars/TensorRT-10.8.0.43.Linux.#x86_64-gnu.cuda-12.8.tar.gz

wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.7.0/tars/TensorRT-10.7.0.23.Linux.x86_64-gnu.cuda-12.6.tar.gz

# Untar
workdir=$(pwd)

tar -xzvf TensorR*

# Move and rename
# tensorrt10.8 for cuda 12.8
#sudo mv TensorRT-10.8.0.43.Linux.x86_64-gnu.cuda-12.8/ /usr/local/
#cd /usr/local/
#sudo mv TensorRT-10.8.0.43.Linux.x86_64-gnu.cuda-12.8/ tensorrt10.8-cuda12.8/
#cd tensorrt10.8-cuda12.8/
#mv TensorRT-10.8.0.43/* .
#rm -r TensorRT-10.8.0.43

# tensorrt10.7 for cuda 12.6
sudo mv TensorRT-10.7.0.23 /usr/local/
cd /usr/local/
sudo mv TensorRT-10.7.0.23 tensorrt10.7-cuda12.6/
cd tensorrt10.7-cuda12.6/

# Add to environment variables
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/tensorrt10.8-cuda12.8/lib

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/tensorrt10.7-cuda12.6/lib

# Add export to bashrc
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/tensorrt10.7-cuda12.6/lib" >> ~/.bashrc
echo "export PATH=/usr/local/tensorrt10.7-cuda12.6/${PATH:+:${PATH}}" >> ~/.bashrc
echo "export TENSORRT_HOME=/usr/local/tensorrt10.7-cuda12.6" >> ~/.bashrc
echo "export nvinfer_LIB_PATH=/usr/local/tensorrt10.7-cuda12.6/lib" >> ~/.bashrc
echo "export TRT_LIB_DIR=/usr/local/tensorrt10.7-cuda12.6/lib" >> ~/.bashrc

#echo "export LD_LIBRARY_PATH=/usr/local/tensorrt10.8-cuda12.8/lib:$LD_LIBRARY_PATH" >> ~/.bashrc

# Install the python wheels for python 3.11
cd python
python3 -m pip install tensorrt-*-cp311-none-linux_x86_64.whl
python3 -m pip install tensorrt_lean-*-cp311-none-linux_x86_64.whl
python3 -m pip install tensorrt_dispatch-*-cp311-none-linux_x86_64.whl

# Run ldconfig
sudo ldconfig

# Clean up folder
cd $workdir
rm -xzvf TensorR*
