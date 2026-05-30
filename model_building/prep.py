from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

# 1. Define Features and Target
# Drop the target class AND the redundant Celsius columns to isolate Kelvin
X = df_hf_copy.drop(['Engine Condition', 'lub oil temp', 'Coolant temp'  ], axis=1)
y = df_hf_copy['Engine Condition']

# 2. Split with Stratification
# 'stratify=y' ensures the 0/1 ratio is preserved in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train size:", X_train.shape, "Test size:", X_test.shape)
print("Class distribution in train:", y_train.value_counts())
print("Class distribution in test:", y_test.value_counts())
