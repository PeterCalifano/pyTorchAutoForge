'''
Script created by PeterC, 12-07-2024, to analyze statistics and performance of Optuna studies
'''
import optuna, os

def GetOptunaStudy(studyName:str, storagePath:str='optuna_db') -> optuna.study.Study:

    # Define storage path
    storagePath = 'sqlite:///{studyPath}.db'.format(studyPath=os.path.join('optuna_db', studyName))

    if not (os.path.exists(storagePath)):
        raise ImportError('File not found at: ', storagePath)
    
    print('Loading study from storage path: ', storagePath)

    # Load study
    studyObj = optuna.load_study(studyName, storage=storagePath)

    return studyObj

def VisualizeStudy_OptimHistory(studyObj:optuna.study.Study) -> None:
    fig = optuna.visualization.plot_optimization_history(studyObj)
    fig.show()

def VisualizeStudy_HyperparamsImportance(studyObj: optuna.study.Study) -> None:
    fig = optuna.visualization.plot_param_importances(studyObj)
    fig.show()

def VisualizeStudy_Timeline(studyObj: optuna.study.Study) -> None:
    fig = optuna.visualization.plot_timeline(studyObj)
    fig.show()

# %% TEST SCRIPT
def main():
    print('\n\n----------------------------------- TEST SCRIPT: AnalyzeOptunaStudy.py -----------------------------------\n')
    # Get study object
    studyName = 'HorizonExtractionEnhancer_deepNNv8_funny-ram-724'
    storagePath = 'optuna_db'
    studyObj = GetOptunaStudy(studyName)

if __name__ == '__main__':
    main()