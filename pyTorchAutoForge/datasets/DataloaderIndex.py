from numpy import isscalar
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

    def __init__(self, trainLoader: DataLoader, 
                 validLoader: DataLoader | None = None, 
                 split_ratio: int | float | tuple = 0.8, 
                 split_seed : int = 42,
                 testLoader: DataLoader | None = None) -> None:
        
        if not(isinstance(trainLoader, DataLoader)):
            raise TypeError('Training dataloader is not of type "DataLoader"!')

        if not(isinstance(validLoader, DataLoader)) and validLoader is not None:
            raise TypeError('Validation dataloader is not of type "DataLoader"!')
        
        if validLoader is not None:
            # Just assign dataloaders
            self.TrainingDataLoader = trainLoader
            self.ValidationDataLoader = validLoader
            self.testLoader = testLoader

        else:
            # Perform random splitting of training data to get validation dataset
            print(f'\033[93mNo validation dataset provided: training dataset automatically split with ratio {split_ratio}\033[0m')

            from torch import Generator

            # Fix generator equal to provided seed
            split_generator_ = Generator().manual_seed(split_seed)
            with_test_dataset = False
            testData = None
            self.testLoader = None

            if isinstance(split_ratio, int | float):

                training_split_fraction = split_ratio
                validation_split_fraction = 1 - split_ratio

                # Split the dataset
                trainingData, validationData = random_split(trainLoader.dataset,
                                                            [training_split_fraction, validation_split_fraction], generator=split_generator_)
                
            elif isinstance(split_ratio, tuple):

                if len(split_ratio) != 3:
                    raise ValueError('split_ratio must be a float | int | a tuple of three floats [train, valid, test].')

                training_split_fraction = split_ratio[0]
                validation_split_fraction = split_ratio[1]
                test_split_fraction = split_ratio[2]

                with_test_dataset = True

                # Split the dataset
                trainingData, validationData, testData = random_split(trainLoader.dataset,
                                                                      [training_split_fraction, validation_split_fraction, test_split_fraction], generator=split_generator_)
            else:
                raise TypeError('split_ratio must be a float | int | a tuple of three floats [train, valid, test].')
            
            # Create dataloaders
            self.TrainingDataLoader = DataLoader(trainingData, batch_size=trainLoader.batch_size, shuffle=True, 
                                                 num_workers=trainLoader.num_workers, drop_last=trainLoader.drop_last)
            
            self.ValidationDataLoader = DataLoader(validationData, batch_size=trainLoader.batch_size, shuffle=True,
                                                   num_workers=trainLoader.num_workers, drop_last=False)
            
            if with_test_dataset and testData is not None:
                self.testLoader = DataLoader(testData, batch_size=trainLoader.batch_size, shuffle=False,
                                             num_workers=0, drop_last=False)
                

    # TODO remove these methods, not necessary in python...
    def getTrainLoader(self) -> DataLoader:
        return self.TrainingDataLoader
    
    def getValidationLoader(self) -> DataLoader:
        return self.ValidationDataLoader