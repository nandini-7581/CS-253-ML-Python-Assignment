import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn import tree

from sklearn.metrics import confusion_matrix
import graphviz

# Load the dataset from train.csv
train_data = pd.read_csv("train.csv")

# Define a function to clean and convert string values to numeric
def clean_and_convert(value):
    try:
        # Remove non-numeric characters and convert to float
        return float(''.join(filter(str.isdigit, str(value))))
    except ValueError:
        # If conversion fails, return NaN
        return float('NaN')

# Apply the clean_and_convert function to the columns containing strings
cols_to_convert = [5, 6, 7]  # Assuming columns 5 through 7 need conversion
for col in cols_to_convert:
    train_data.iloc[:, col] = train_data.iloc[:, col].apply(clean_and_convert)







# Extract features and target variable
X = train_data.iloc[:, 4:8]   # Features in columns 4 through 7
Y = train_data.iloc[:, 8]     # Target variable in column 8

#unique_classes = Y.unique()
#print(unique_classes)


cf = DecisionTreeClassifier()
cf.fit(X, Y)

# Load the test dataset from test.csv
test_data = pd.read_csv("test.csv")

# Preprocess the test data (apply the same preprocessing steps as the training data)
# For example, if you need to convert string values to numeric:
for col in cols_to_convert:
    test_data.iloc[:, col] = test_data.iloc[:, col].apply(clean_and_convert)

# Extract features from the test data
X_test = test_data.iloc[:, 4:8]  # Assuming features are in columns 4 through 7

# Predict target variable for the test data
Ypred = cf.predict(X_test)

# Print predicted values
print("Predicted values for test data:")
print(Ypred)
# Add predicted values to the test_data dataframe
test_data['Predicted_Values'] = Ypred

# Save the dataframe with predicted values back to a CSV file
test_data.to_csv("test_with_predictions.csv", index=False)



#cmat = confusion_matrix(Y, Ypred)

# Print confusion matrix
#print('Confusion matrix:')
#print(cmat)

# Print true and predicted values side by side
#print('True Values    Predicted Values')
#for true_val, pred_val in zip(Y, Ypred):
 # print(f'{true_val}             {pred_val}')

# Plot decision tree
#decPlot = plot_tree(decision_tree=cf, feature_names=["Criminal_Case", "total_assets", "liabilities", "state"],
 #                   class_names=["8th_pass", "10th_Pass", "12th_pass","graduate_professional","post_graduate","graduate","Others","Doctorate","literate","5th_pass"], filled=True, precision=4, rounded=True)

#text_representation = tree.export_text(cf, feature_names=["Criminal_Case", "total_assets", "liabilities", "state"])
#print(text_representation)

#dot_data = tree.export_graphviz(cf, out_file=None,
 #                               feature_names=["Criminal_Case", "total_assets", "liabilities", "state"],
  #                              class_names=["8th_pass", "10th_Pass", "12th_pass","graduate_professional","post_graduate","graduate","Others","Doctorate","literate","5th_pass"],
   #                             filled=True, rounded=True,
    #                            special_characters=True)
#graph = graphviz.Source(dot_data)
