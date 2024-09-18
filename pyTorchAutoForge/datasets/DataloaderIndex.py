from torch.utils.data import DataLoader, random_split
from typing import Optional

# %%  Data loader indexer class - PeterC - 23-07-2024
class DataloaderIndex:
    '''
    Class to index dataloaders for training and validation datasets. Performs splitting if validation loader is not provided.
    Created by PeterC, 23-07-2024
    '''
    def __init__(self, trainLoader:DataLoader, validLoader:Optional[DataLoader] = None) -> None:
        if not(isinstance(trainLoader, DataLoader)):
            raise TypeError('Training dataloader is not of type "DataLoader"!')

        if validLoader is None:
            # Perform random splitting of training data to get validation dataset
            print('No validation dataset provided: training dataset automatically split with ratio 80/20')
            trainingSize = int(0.8 * len(trainLoader.dataset))
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

    def getTrainloader(self) -> DataLoader:
        return self.TrainingDataLoader
    
    def getValidationLoader(self) -> DataLoader:
        return self.ValidationDataLoader