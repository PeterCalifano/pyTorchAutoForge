# Quick note on the model building module

## The adapter-backbone-model design

- A generic model is assumed to be a combination of a **backbone** model extracting features from the input (whatever this is, presumably a 2D volume) and a **head**, that can be either single or multi head (see available classes). The key idea anyhow, is that they must be able to be stacked in a `Sequential` object.
- The **backbone** may have an input adapter specified if one wants to add extra layers before the backbone (which for default cases is a pre-existing model from torchvision or anywhere else, but must be implemented manually).
- Each of these pieces shall have a configuration object that inherits from a base configuration (all shall be linked to `BaseConfigClass`) so that it can be loaded from yml file through dacite if needed, and shall be a dataclass.
- The backbone is composed of the adapter and a class that inherits from `FeatureExtractor`. This essentially enables the construction of a factory function `FeatureExtractorFactory` that works by building it through singledispatch.
  - The same holds for the adapters, `BaseInputAdapter` and `InputAdapterFactory`.
