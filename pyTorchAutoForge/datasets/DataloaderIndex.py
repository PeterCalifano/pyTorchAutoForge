from torch.utils.data import DataLoader, random_split
from typing import Optional

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
    def __init__(self, trainLoader:DataLoader, validLoader:Optional[DataLoader] = None, split_ratio:int = 0.8) -> None:
        if not(isinstance(trainLoader, DataLoader)):
            raise TypeError('Training dataloader is not of type "DataLoader"!')

        if validLoader is None:
            # Perform random splitting of training data to get validation dataset
            print('No validation dataset provided: training dataset automatically split with ratio 80/20')
            trainingSize = int(split_ratio * len(trainLoader.dataset))
            validationSize = len(trainLoader.dataset) - trainingSize

            # Split the dataset
            trainingData, validationData = random_split(trainLoader.dataset, [trainingSize, validationSize])

            # Create dataloaders
            self.TrainingDataLoader = DataLoader(trainingData, batch_size=trainLoader.batch_size, shuffle=True, 
                                                 num_workers=trainLoader.num_workers, drop_last=trainLoader.drop_last)
            self.ValidationDataLoader = DataLoader(validationData, batch_size=trainLoader.batch_size, shuffle=True,
                                                   num_workers=trainLoader.num_workers, drop_last=False)
        else:

            self.TrainingDataLoader = trainLoader

            if not(isinstance(validLoader, DataLoader)):
                raise TypeError('Validation dataloader is not of type "DataLoader"!')
            
            self.ValidationDataLoader = validLoader

    def getTrainLoader(self) -> DataLoader:
        return self.TrainingDataLoader
    
    def getValidationLoader(self) -> DataLoader:
        return self.ValidationDataLoader