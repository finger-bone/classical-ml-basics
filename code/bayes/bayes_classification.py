import numpy as np

class NaiveBayesClassifier:
    def __init__(self, feature_types):
        self.feature_types = feature_types  # List indicating type ('continuous'/'discrete') for each feature
        self.classes = None
        self.prior = {}
        self.params = []  # For storing feature parameters (mean/std for continuous, probabilities for discrete)
    
    def fit(self, X, y):
        self.classes = np.unique(y)
        n_features = X.shape[1]
        self.params = [{} for _ in range(n_features)]
        
        # Calculate prior probabilities
        for cls in self.classes:
            self.prior[cls] = np.mean(y == cls)
        
        # Calculate likelihood parameters for each feature and class
        for i in range(n_features):
            feature_type = self.feature_types[i]
            for cls in self.classes:
                X_cls = X[y == cls, i]
                if feature_type == 'discrete':
                    # Discrete feature: calculate probability of each value
                    values, counts = np.unique(X_cls, return_counts=True)
                    self.params[i][cls] = {v: c/len(X_cls) for v, c in zip(values, counts)}
                else:
                    # Continuous feature: calculate mean and standard deviation
                    mean = np.mean(X_cls)
                    std = np.std(X_cls, ddof=1)  # Unbiased estimator
                    self.params[i][cls] = (mean, std)
    
    def predict(self, X):
        predictions = []
        for sample in X:
            max_log_prob = -np.inf
            best_class = None
            for cls in self.classes:
                log_prob = np.log(self.prior[cls])
                valid = True
                for i in range(len(self.feature_types)):
                    feature_val = sample[i]
                    param = self.params[i][cls]
                    if self.feature_types[i] == 'discrete':
                        # Handle unseen discrete values with 0 probability
                        prob = param.get(feature_val, 0.0)
                        if prob == 0:
                            valid = False
                            break
                        log_prob += np.log(prob)
                    else:
                        # Gaussian probability density
                        mean, std = param
                        if std == 0:
                            if feature_val != mean:
                                valid = False
                                break
                        else:
                            log_prob -= 0.5 * (np.log(2 * np.pi) + 2 * np.log(std)) + ((feature_val - mean) ** 2) / (2 * std ** 2)
                if valid and log_prob > max_log_prob:
                    max_log_prob = log_prob
                    best_class = cls
            predictions.append(best_class if best_class is not None else self.classes[0])
        return np.array(predictions)

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X = iris.data 
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

feature_types = ['continuous'] * X.shape[1]

nb_classifier = NaiveBayesClassifier(feature_types=feature_types)
nb_classifier.fit(X_train, y_train)

y_pred = nb_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")