import torch
from pathlib import Path
from pyTorchAutoForge.model_building.backbones.efficient_net import EfficientNetConfig, FeatureExtractorFactory

# Export efficient net to ONNX
def test_efficientnet_basic_backbone_export(tmp_path: Path):
    # Create configuration
    cfg = EfficientNetConfig(
        model_name='b0',
        input_resolution=(224, 224),
        pretrained=True,
        output_size=10,
        remove_classifier=True,
        device='cpu',
        input_channels=3,
        output_type='last',
    )

    backbone_last = FeatureExtractorFactory(cfg)
    print(backbone_last)
    backbone_last.eval()

    # Batch size of 1, 3 channels, 224x224 image
    dummy_input = torch.randn(1, 3, 224, 224)

    # Define model exorter from api module
    print("Exporting model to ONNX format...")
    from pyTorchAutoForge.api.onnx import ModelHandlerONNx

    onnx_handler = ModelHandlerONNx(model=backbone_last,
                                    dummy_input_sample=dummy_input,
                                    opset_version=18,
                                    onnx_export_path=str(tmp_path))
    
    # Test export with torch legacy pipeline
    model_export_name = "efficientnet_lastfeats_b0_legacy.onnx"
    
    legacy_export_path = onnx_handler.torch_export(onnx_model_name=model_export_name, enable_verbose=True)
    assert Path(legacy_export_path).is_file()

    # Test export with torch dynamo pipeline # BUG torch_dynamo_export currently fails
    model_export_name_dynamo = "efficientnet_lastfeats_b0_dynamo.onnx"
    dynamo_export_path = onnx_handler.torch_dynamo_export(onnx_model_name=model_export_name_dynamo, enable_verbose=True)
    assert Path(dynamo_export_path).is_file()

def test_efficientnet_intermediatefeats_backbone_export():
    pass

# MANUAL TESTING
if __name__ == "__main__":
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmp_dir:
        test_efficientnet_basic_backbone_export(Path(tmp_dir))
    #test_efficientnet_intermediatefeats_backbone_export()
    print("Tests completed.")
