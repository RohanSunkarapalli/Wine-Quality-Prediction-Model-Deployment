# ML Model Deployment and Testing

## Overview

The project aims to train a machine learning model on a tabular dataset: UCI Wine Quality, and deploy locally using development tools, including FastAPI and uvicorn. The project aims to thoroughly understand the packaging ML model and deployment where users can interact and take advantage of it. The crucial part of the framework is deploying the model with endpoints.


## Instructions

### Libraries

- fastapi
- scikit-learn
- NumPy
- pandas
- matplotlib.pyplot
- seaborn
- uvicorn
- joblib
- pydantic

### Files:

- data/raw: white wine and red wine dataset.
- models/gradient_boosting: Robust scaler for features and target columns. Trained model packaged with joblib.

### Execution

- file: modeling.ipynb
    - The Jupyter Notebook has the necessary libraries for a smooth run.
    - Run all cells in the notebook to process the following in sequential order:
        - Data loading.
        - Data preprocessing.
        - Create 5-fold cross-validation using scikit-learn's StratifiedKFold module. Fold 6 is reserved for out-of-box testing.
        - Design and implement model architecture.
        - Loop over the folds to train and validate to investigate optimum hyperparameters. Next, generate performance metrics of cross-validation.
        - Use the optimum hyperparameters to train the model with all samples in the training dataset and make predictions for the test dataset. Next, generate a report on the train and test the mean square error.
- file: deploy_test.ipynb
    - Change the directory to src.
    - Run the deploy.py model to deploy and test the trained model.
    - Pass sample data in the post request and execute to get wine quality prediction.

## Contributing

Feel free to contribute to the project by opening issues or submitting pull requests. Please follow coding standards and provide clear documentation for any changes.

## License

This project is licensed under the [MIT License] (LICENSE).
