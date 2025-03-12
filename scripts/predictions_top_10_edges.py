import numpy as np
import pandas as pd
import os
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneOut
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import clone
from sklearn.feature_selection import RFE
from sklearn.inspection import permutation_importance

# Define path to the file containing Subject ID and cognitive scores
# Replace this with the path to your cognitive scores CSV file
path_to_info_file = "<path_to_cognitive_scores_csv>"  

# Root directory containing similarity files
# Replace this with the path to your similarity files directory
root_dir_similarities = "<path_to_similarity_files>" 

# Initialize data set
X = []
Y = []

# Define cognitive score columns
cognitive_score_columns = ['CDRSB', 'ADAS11', 'ADAS13', 'ADASQ4', 'MMSE', 'RAVLT_immediate', 
                           'RAVLT_learning', 'RAVLT_perc_forgetting']

# Read the cognitive score file
df = pd.read_csv(path_to_info_file, sep=',', header=0)

# Filter rows with missing cognitive scores
df = df.dropna(subset=cognitive_score_columns)

# Initialize data arrays
X, Y, DX_list = [], [], []
regions_cache = {}

# Process each subject's data
for index, row in df.iterrows():
    subject_id = row['SubjectID']
    found_file = False
    
    # Traverse directories and load similarity files
    for root, dirs, files in os.walk(root_dir_similarities):
        expected_file = f"{subject_id}_similarity.csv"
        if expected_file in files:
            found_file = True
            file_path = os.path.join(root, expected_file)
            
            # Read the similarity file
            similarity = pd.read_csv(file_path)
            regions = np.unique(similarity[['Region 1', 'Region 2']].values.flatten())
            
            # Cache regions to avoid recomputation
            if subject_id not in regions_cache:
                regions_cache[subject_id] = regions

            edge_matrix = np.zeros((len(regions), len(regions)))
            
            for _, sim_row in similarity.iterrows():
                idx1 = np.where(regions == sim_row['Region 1'])[0][0]
                idx2 = np.where(regions == sim_row['Region 2'])[0][0]
                edge_matrix[idx1, idx2] = sim_row['Similarity']
                edge_matrix[idx2, idx1] = sim_row['Similarity']

            upper_triangle_indices = np.triu_indices(len(regions), k=1)
            edge_matrix_vector = edge_matrix[upper_triangle_indices]

            X.append(edge_matrix_vector)

            # Append cognitive scores
            list_cog = [row[col] for col in cognitive_score_columns]
            Y.append(list_cog)
            DX_list.append(row['DX'])
            break  # Stop searching after finding the file

    if not found_file:
        print(f"Warning: Similarity file for subject {subject_id} not found.")

# Convert X, Y, and DX to numpy arrays
X = np.array(X)
Y = np.array(Y)
DX_values = np.array(DX_list)

# Ensure that X and Y have consistent dimensions
if len(X) != len(Y):
    raise ValueError(f"Inconsistent data dimensions: len(X)={len(X)}, len(Y)={len(Y)}")

# Define the best parameters directly in SVR
best_svr = SVR(kernel='rbf', C=10, gamma=0.001, degree=2, epsilon=0.2)

# Initialize the pipeline with MinMaxScaler and SVR
pipeline = Pipeline([
    ('scaler', MinMaxScaler(feature_range=(0, 1))),
    ('svr', MultiOutputRegressor(best_svr))
])

# Fit the pipeline on the full dataset
pipeline.fit(X, Y)
best_model = pipeline

# Prepare lists to store true and predicted values
Y_true = []
Y_pred = []

# Create Leave-One-Out Cross-Validation
cv = LeaveOneOut()

# Enumerate splits of the data
for train_ix, test_ix in cv.split(X):
    # Split data
    X_train, X_test = X[train_ix], X[test_ix]
    y_train, y_test = Y[train_ix], Y[test_ix]

    # Clone the best model to ensure a fresh model for each iteration
    model = clone(best_model)

    # Fit model
    model.fit(X_train, y_train)

    # Evaluate model
    y_hat = model.predict(X_test)

    # Store the true and predicted values
    Y_true.append(y_test[0])
    Y_pred.append(y_hat[0])

# Convert lists to NumPy arrays
Y_true = np.array(Y_true)
Y_pred = np.array(Y_pred)

# Calculate R² score
r2_SVR = r2_score(Y_true, Y_pred)
print(f'Multi-Output SVR R² Score: {r2_SVR}')

# Loop over each cognitive score to create separate files with the predicted values vs. true values for the plots
output_dir = "<path_to_output_directory>"  # Replace with the path to save results
os.makedirs(output_dir, exist_ok=True)

for i, cognitive_score in enumerate(cognitive_score_columns):
    # Extract the ith element of Y_true and Y_pred for each cognitive score and each patient
    element_Y_true = Y_true[:, i]
    element_Y_pred = Y_pred[:, i]

    # Create a new DataFrame with the desired columns
    result_df = pd.DataFrame({
        'Actual': element_Y_true,
        'Predicted': element_Y_pred,
        'DX': DX_values
    })
    
    # Define the output file path based on the cognitive score name
    output_path = os.path.join(output_dir, f"{cognitive_score}_ytrue_ypred.csv")
    
    # Save the new dataframe to a CSV file
    result_df.to_csv(output_path, index=False)
    
    # Print a success message for each file
    print(f"CSV file created for {cognitive_score}: {output_path}")

# Now, using permutation importance, determine top 10 most important edges for predicting each cognitive score
importance_output_dir = os.path.join(output_dir, "top_10_edges")
os.makedirs(importance_output_dir, exist_ok=True)

num_regions = 200

# Loop over each cognitive score
for i, cognitive_score in enumerate(cognitive_score_columns):
    # Set up a single-output SVR model for the specific cognitive score
    single_target_model = Pipeline([
        ('scaler', MinMaxScaler(feature_range=(0, 1))),
        ('svr', SVR(kernel='rbf', C=10, gamma=0.001, degree=2, epsilon=0.2))
    ])

    # Fit the model on all data for this specific cognitive score
    single_target_model.fit(X, Y[:, i])

    # Compute permutation importance for the model on this cognitive score
    perm_importance = permutation_importance(single_target_model, X, Y[:, i], n_repeats=5, random_state=0)

    # Get indices of the top 10 most important features
    top_10_indices = np.argsort(perm_importance.importances_mean)[-10:]

    # Initialize a 200x200 matrix with all entries as 0
    importance_matrix = np.zeros((num_regions, num_regions))

    # Convert the vector indices to matrix positions (upper triangular)
    upper_triangle_indices = np.triu_indices(num_regions, k=1)
    for idx in top_10_indices:
        # Identify row and column from the vector index for the top feature
        row, col = upper_triangle_indices[0][idx], upper_triangle_indices[1][idx]
        importance_matrix[row, col] = 1
        importance_matrix[col, row] = 1  # Symmetrical update

    # Save the matrix as a CSV file for the cognitive score
    output_path = os.path.join(importance_output_dir, f"{cognitive_score}_top_10_edges.csv")
    np.savetxt(output_path, importance_matrix, delimiter=",")

    print(f"Matrix saved for {cognitive_score} at {output_path}")
