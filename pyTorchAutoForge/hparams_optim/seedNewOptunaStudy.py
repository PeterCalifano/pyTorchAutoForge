import optuna
import os

# TODO rework and generalize script

def clone_study(to_study_name: str, from_study_name: str, from_database_path: str, to_database_path: str = None):
    """Function cloning an optuna study from one database to another. Current version assumes that the study name and the database name are the same.

    Args:
        to_study_name (str): _description_
        from_study_name (str): _description_
        from_database_path (str): _description_
        to_database_path (str, optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if to_study_name == "":
        raise ValueError("to_study_name must be specified.")
    
    # Copy storage name if not specified
    if to_database_path is None:
        to_database_path = 'sqlite:///{studyPathAndName}.db'.format(
            studyPathAndName=os.path.join(from_database_path, to_study_name))
    else:
        to_database_path = 'sqlite:///{studyPathAndName}.db'.format(
            studyPathAndName=os.path.join(to_database_path, to_study_name))

    # Create study clone copying everything with different name

    study = optuna.copy_study(
        to_study_name=to_study_name,
        to_storage=to_database_path,
        from_study_name = from_study_name,
        from_storage='sqlite:///{studyPathAndName}.db'.format(
            studyPathAndName=os.path.join(from_database_path, from_study_name)),
    )
    
    return study



def main():

    # UNIT TEST
    print("----------------------------------- UNIT TEST: clone_study function -----------------------------------")

    hostname = os.uname().nodename
    if hostname == 'peterc-desktopMSI':
        OPTUNA_BEST_MODELS_PATH = '/media/peterc/6426d3ea-1f91-40b7-93ab-7f00d034e5cd/optuna_storage/optuna_trials_best_models'
        OPTUNA_DB_PATH = '/media/peterc/6426d3ea-1f91-40b7-93ab-7f00d034e5cd/optuna_storage'

    elif hostname == 'peterc-recoil':
        OPTUNA_BEST_MODELS_PATH = './optuna_storage/optuna_trials_best_models'
        OPTUNA_DB_PATH = './optuna_storage'

    new_study_name = "fullDiskConvNet_HyperParamsOptim_CentroidOnly_GaussNoiseBlurShift_V3"
    from_study_name = "fullDiskConvNet_HyperParamsOptim_CentroidOnly_5000images_V2"

    clone_study(new_study_name, from_study_name, from_database_path=OPTUNA_DB_PATH, to_database_path =None)

if __name__ == '__main__':
    main()