## Project1: SVM Protein Classification

### Environment Setup:
1. **Python Environment:** Ensure you have Python installed (preferably Python 3.x).
2. **Dependencies:** Install required dependencies using pip:

```
pip install numpy pandas scikit-learn biopython
```

### Code Flow:
1. **Data Preprocessing:**
- The code preprocesses the data, loading protein structure diagrams and sequence information.
- If the `--ent` flag is provided, the data is loaded from a file using a feature engineering function `feature_extraction()` from fea.py. Otherwise, it loads from pre-existing files.
- The code reads a CAST file containing protein sequence information and a Numpy array containing diagrams.

2. **Model Initialization:**
- Three types of models are supported and need implementing: Support Vector Machine (SVM), Linear SVM, and Logistic Regression (LR).
- You can specify the model type using the `--model_type` argument. Options are `'svm'`, `'linear_svm'`, and `'lr'`.
- For SVM models, you can choose the kernel type (`--kernel`) from `'linear'`, `'poly'`, `'rbf'`, or `'sigmoid'`.
- Regularization parameter `C` can be set using the `--C` argument.

3. **Training and Evaluation:**
- The code trains the selected model on the training data and evaluates its performance on both training and test datasets.
- It partitions the dataset into training and testing sets for each task.
- The model's accuracy is printed for each dataset.

### Filling in the Blanks:
1. **LRModel Class:**
- Fill in the initialization, training, and evaluation methods for the Logistic Regression model.

2. **LinearSVMModel Class:**
- Implement the initialization method for the Linear SVM model.

3. **Test Data Generation:**
- Complete the generation of test data by complementing the train data. Ensure correct reading positions for test data.

### Running the Script:
- Execute the script `main.py` from the command line.
- You can provide arguments to customize the model type, kernel type, regularization parameter, and data loading method.

### Example Command:
```
python main.py --model_type svm --kernel rbf --C 1.0
```

### Experimental Requirements:

1. **Complete Implementation of Protein Classification (Data Loading) - 3 points**
    - Ensure the code effectively reads and preprocesses protein structure data and sequences.

2. **Comparison of Linear SVM with Other Machine Learning Methods (e.g., LR) - 3 points**
    - Implement Linear SVM model and LR model.
    - Analyze and compare the performance of SVM with other methods in terms of accuracy and computational efficiency.

3. **Analysis and Discussion on the Impact of SVM Kernel Functions and Regularization Coefficients - 2 points**
    - Investigate the effects of different SVM kernel functions on classification performance.
    - Analyze how varying the regularization coefficient (`C`) affects the model's performance and generalization.

4. **Feature Engineering: - 2 points**
    - Provide insights on extracting useful features from protein structure data or utilizing feature selection methods to reduce dimensionality.
    - Modify the feature extraction function for the final experimental analysis.
  
**Total Score: 10 points**

### Submission:
Submit the code along with an experimental report (up to 4 pages, No Involution) to the e-learning platform.

### Note:
- Ensure you now are in 'pj1' path.
- Adjust the arguments as per your requirements for experimentation.
- Ensure the experimental report includes detailed explanations of the implemented code, experimental setup, results, analysis, and discussions based on the specified requirements.