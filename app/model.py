# import pathlib
# from fastai.vision.all import *

# # Temporary redirect PosixPath to WindowsPath
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

# def load_model():
#     model_path = Path('saved_model/pneumonia_classification_model.pkl')
    
#     # Ensure the path is a string
#     model_path_str = str(model_path)
    
#     # Debugging: Print the path to ensure it's correct
#     print(f"Loading model from: {model_path_str}")
    
#     # Load the model
#     model = load_learner(model_path_str)
    
#     # Return the model
#     return model

# # Restore PosixPath
# pathlib.PosixPath = temp


# Streamlit try
from fastai.learner import load_learner

def load_model():
    model_path = 'saved_model/pneumonia_classification_model.pkl'
    
    # Ensure the path is a string
    model_path_str = str(model_path)
    
    # Debugging: Print the path to ensure it's correct
    print(f"Loading model from: {model_path_str}")
    
    # Load the model
    model = load_learner(model_path_str)
    
    # Return the model
    return model

