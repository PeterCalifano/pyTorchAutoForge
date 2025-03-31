from torch.utils.data import DataLoader, random_split
from math import floor 
# Removed Optional as it is deprecated in Python 3.10; using "| None" instead

# %%  Data loader indexer class - PeterC - 23-07-2024
class DataloaderIndex:
    """
    DataloaderIndex class to index dataloaders for training and validation datasets. 
    This class performs splitting of the training dataset if a separate validation loader is not provided.
    Attributes:
        TrainingDataLoader (DataLoader): DataLoader for the training dataset.
        ValidationDataLoader (DataLoader): DataLoader for the validation dataset.
    Methods:
        __init__(trainLoader: DataLoader, validLoader: Optional[DataLoader] = None) -> None:
            Initializes the DataloaderIndex with the provided training and optional validation dataloaders.
            If no validation dataloader is provided, splits the training dataset into training and validation datasets.
        getTrainLoader() -> DataLoader:
            Returns the DataLoader for the training dataset.
        getValidationLoader() -> DataLoader:
            Returns the DataLoader for the validation dataset.
    """
    def __init__(self, trainLoader : DataLoader, validLoader:DataLoader | None = None, split_ratio : int | float = 0.8) -> None:
        if not(isinstance(trainLoader, DataLoader)):
            raise TypeError('Training dataloader is not of type "DataLoader"!')

        if not(isinstance(validLoader, DataLoader)) and validLoader is not None:
            raise TypeError('Validation dataloader is not of type "DataLoader"!')
        
        if validLoader is not None:
            # Just assign dataloaders
            self.TrainingDataLoader = trainLoader
            self.ValidationDataLoader = validLoader
        else:
            # Perform random splitting of training data to get validation dataset
            print(f'\033[93mNo validation dataset provided: training dataset automatically split with ratio {split_ratio}\033[0m')

            training_size = floor(split_ratio * trainLoader.__len__())
            validation_size = trainLoader.__len__() - training_size

            # Split the dataset
            trainingData, validationData = random_split(trainLoader.dataset, [training_size, validation_size])

            # Create dataloaders
            self.TrainingDataLoader = DataLoader(trainingData, batch_size=trainLoader.batch_size, shuffle=True, 
                                                 num_workers=trainLoader.num_workers, drop_last=trainLoader.drop_last)
            
            self.ValidationDataLoader = DataLoader(validationData, batch_size=trainLoader.batch_size, shuffle=True,
                                                   num_workers=trainLoader.num_workers, drop_last=False)

    # TODO remove these methods, not necessary in python...
    def getTrainLoader(self) -> DataLoader:
        return self.TrainingDataLoader
    
    def getValidationLoader(self) -> DataLoader:
        return self.ValidationDataLoader