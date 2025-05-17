from numpy import isscalar
from torch.utils.data import DataLoader, random_split
from math import floor 

# %%  Data loader indexer class - PeterC - 23-07-2024
# TODO add support for RepeatedKfolder sampler from sklearn
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

    # TODO modify to accept datasets directly and a combination. If dataset is input, use default specifications for dataloader

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
            self.TrainingDataLoader: DataLoader = trainLoader
            self.ValidationDataLoader: DataLoader = validLoader
            self.testingDataLoader: DataLoader | None = testLoader

        else:
            # Perform random splitting of training data to get validation dataset
            print(f'\033[93mNo validation dataset provided: training dataset automatically split with ratio {split_ratio}\033[0m')

            from torch import Generator

            # Fix generator equal to provided seed
            split_generator_ = Generator().manual_seed(split_seed)
            with_test_dataset = False
            testData = None
            self.testingDataLoader = None

            if isinstance(split_ratio, int | float):

                training_split_fraction = split_ratio
                validation_split_fraction = 1 - split_ratio

                # Split the dataset
                trainingData, validationData = random_split(trainLoader.dataset,
                                                            [training_split_fraction, validation_split_fraction], generator=split_generator_)
                
            elif isinstance(split_ratio, tuple):

                if len(split_ratio) != 3:
                    raise ValueError('split_ratio must be a float | int | a tuple of three floats [train, valid, test].')

                if any(ratio < 0 or ratio > 1 for ratio in split_ratio):
                    raise ValueError('Invalid split ratios: must be between 0 and 1.')

                if sum(split_ratio) != 1:
                    print(f'\033[93mWarning: split_ratio does not sum to 1.0, but to {sum(split_ratio)}. Validation split will be overridden.\033[0m')
                    
                    training_split_fraction = split_ratio[0]
                    test_split_fraction = split_ratio[2]
                    validation_split_fraction = 1 - training_split_fraction - test_split_fraction

                else:
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
                self.testingDataLoader = DataLoader(testData, batch_size=trainLoader.batch_size, shuffle=False,
                                             num_workers=0, drop_last=False)
                

    # TODO remove these methods, not necessary in python...
    def getTrainLoader(self) -> DataLoader:
        return self.TrainingDataLoader
    
    def getValidationLoader(self) -> DataLoader:
        return self.ValidationDataLoader