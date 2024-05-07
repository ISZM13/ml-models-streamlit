import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the cleaned penguins data
penguins = pd.read_csv('penguin/penguins_cleaned.csv')

# Ordinal feature encoding
encode = ['sex', 'island']
for col in encode:
    dummy = pd.get_dummies(penguins[col], prefix=col)
    penguins = pd.concat([penguins, dummy], axis=1)
    del penguins[col]

# Define target mapping
target_mapper = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}
def target_encode(val):
    return target_mapper[val]

# Apply target encoding
penguins['species'] = penguins['species'].apply(target_encode)

# Separating X and y
X = penguins.drop('species', axis=1)
y = penguins['species']

# Build and train the Random Forest classifier
clf = RandomForestClassifier()
clf.fit(X, y)

# Save the trained model using joblib
joblib.dump(clf, 'penguin/penguins_clf.joblib')
