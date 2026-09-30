import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from src.preprocessing import CreditDataPipeline

def sigmoid(z: np.ndarray) -> np.ndarray:
    z_clipped = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z_clipped))

def roc_auc_score_numpy(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Calculates ROC-AUC score using trapezoidal rule on probability rank ordering."""
    order = np.argsort(-y_proba)
    y_true_sorted = y_true[order]
    
    n_pos = np.sum(y_true_sorted == 1)
    n_neg = np.sum(y_true_sorted == 0)
    
    if n_pos == 0 or n_neg == 0:
        return 0.5
        
    tpr = np.cumsum(y_true_sorted == 1) / n_pos
    fpr = np.cumsum(y_true_sorted == 0) / n_neg
    
    tpr = np.concatenate(([0.0], tpr))
    fpr = np.concatenate(([0.0], fpr))
    
    auc = np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0)
    return float(np.clip(auc, 0.0, 1.0))

def calculate_ks_statistic(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Calculates Kolmogorov-Smirnov (KS) statistic."""
    order = np.argsort(-y_proba)
    y_true_sorted = y_true[order]
    
    n_pos = np.sum(y_true_sorted == 1)
    n_neg = np.sum(y_true_sorted == 0)
    
    if n_pos == 0 or n_neg == 0:
        return 0.0
        
    tpr = np.cumsum(y_true_sorted == 1) / n_pos
    fpr = np.cumsum(y_true_sorted == 0) / n_neg
    
    ks_stat = np.max(np.abs(tpr - fpr))
    return float(ks_stat)

def calculate_gini_coefficient(roc_auc: float) -> float:
    return float(2.0 * roc_auc - 1.0)

class LogisticRegressionNumPy:
    """Logistic Regression implemented in pure NumPy with L2 Regularization."""
    
    def __init__(self, lr: float = 0.05, n_iter: int = 500, l2_reg: float = 0.01, random_state: int = 42):
        self.lr = lr
        self.n_iter = n_iter
        self.l2_reg = l2_reg
        self.random_state = random_state
        self.weights = None
        self.bias = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegressionNumPy':
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        self.weights = np.random.normal(0, 0.01, size=n_features)
        self.bias = 0.0
        
        for _ in range(self.n_iter):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = sigmoid(linear_model)
            
            dw = (1.0 / n_samples) * np.dot(X.T, (y_predicted - y)) + (self.l2_reg / n_samples) * self.weights
            db = (1.0 / n_samples) * np.sum(y_predicted - y)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        linear_model = np.dot(X, self.weights) + self.bias
        p1 = sigmoid(linear_model)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    @property
    def feature_importances_(self) -> np.ndarray:
        return np.abs(self.weights)

class DecisionNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    @property
    def is_leaf_node(self):
        return self.value is not None

class DecisionTreeNumPy:
    def __init__(self, max_depth: int = 6, min_samples_split: int = 5, max_features: float = 0.8):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.root = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTreeNumPy':
        self.n_features_tot = X.shape[1]
        self.n_features = max(1, int(self.max_features * self.n_features_tot))
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _gini(self, y):
        if len(y) == 0: return 0.0
        p = np.mean(y)
        return 1.0 - (p**2 + (1.0 - p)**2)

    def _best_split(self, X, y):
        best_gain = -1.0
        split_idx, split_thresh = None, None
        feat_indices = np.random.choice(self.n_features_tot, self.n_features, replace=False)
        parent_gini = self._gini(y)

        for feat_idx in feat_indices:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            if len(thresholds) > 10:
                thresholds = np.percentile(X_column, np.linspace(10, 90, 8))

            for thresh in thresholds:
                left_mask = X_column <= thresh
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                left_gini = self._gini(y[left_mask])
                right_gini = self._gini(y[right_mask])
                p_left = np.sum(left_mask) / len(y)
                gain = parent_gini - (p_left * left_gini + (1 - p_left) * right_gini)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = thresh

        return split_idx, split_thresh

    def _build_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = np.mean(y) if len(y) > 0 else 0.0
            return DecisionNode(value=leaf_value)

        feat_idx, thresh = self._best_split(X, y)
        if feat_idx is None:
            return DecisionNode(value=np.mean(y))

        left_mask = X[:, feat_idx] <= thresh
        right_mask = ~left_mask

        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return DecisionNode(feature=feat_idx, threshold=thresh, left=left_child, right=right_child)

    def _traverse_tree(self, x, node):
        if node.is_leaf_node:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p1 = np.array([self._traverse_tree(x, self.root) for x in X])
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

class RandomForestNumPy:
    """Random Forest Classifier implemented in pure NumPy."""
    
    def __init__(self, n_trees: int = 25, max_depth: int = 6, min_samples_split: int = 5, random_state: int = 42):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.trees = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestNumPy':
        np.random.seed(self.random_state)
        n_samples = X.shape[1]
        self.trees = []

        for _ in range(self.n_trees):
            idxs = np.random.choice(len(X), len(X), replace=True)
            X_sample, y_sample = X[idxs], y[idxs]
            tree = DecisionTreeNumPy(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        tree_preds = np.array([t.predict_proba(X)[:, 1] for t in self.trees])
        p1 = np.mean(tree_preds, axis=0)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

class GradientBoostingNumPy:
    """Gradient Boosted Decision Trees implemented in pure NumPy."""
    
    def __init__(self, n_estimators: int = 20, learning_rate: float = 0.1, max_depth: int = 4, random_state: int = 42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []
        self.base_pred = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GradientBoostingNumPy':
        np.random.seed(self.random_state)
        p = np.mean(y)
        self.base_pred = np.log(p / (1.0 - p + 1e-7))
        
        F = np.full(len(y), self.base_pred)
        self.trees = []

        for _ in range(self.n_estimators):
            probabilities = sigmoid(F)
            residuals = y - probabilities  # Gradient of log-loss
            
            tree = DecisionTreeNumPy(max_depth=self.max_depth, min_samples_split=5)
            tree.fit(X, residuals)
            
            update = tree.predict_proba(X)[:, 1]
            F += self.learning_rate * update
            self.trees.append(tree)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        F = np.full(len(X), self.base_pred)
        for tree in self.trees:
            F += self.learning_rate * tree.predict_proba(X)[:, 1]
        p1 = sigmoid(F)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

class CreditRiskPipelineWrapper:
    """Wraps preprocessor and classifier into a unified pipeline interface."""
    
    def __init__(self, classifier_name: str, classifier_obj):
        self.classifier_name = classifier_name
        self.classifier = classifier_obj
        self.pipeline = CreditDataPipeline()

    def fit(self, df_train: pd.DataFrame, y_train: pd.Series) -> 'CreditRiskPipelineWrapper':
        X_trans = self.pipeline.fit_transform(df_train)
        self.classifier.fit(X_trans, y_train.values)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X_trans = self.pipeline.transform(df)
        return self.classifier.predict_proba(X_trans)

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        X_trans = self.pipeline.transform(df)
        return self.classifier.predict(X_trans, threshold=threshold)

class CreditRiskModelTrainer:
    """Trains and benchmarks Logistic Regression, Random Forest, and Gradient Boosting models."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {
            'Logistic Regression': LogisticRegressionNumPy(lr=0.08, n_iter=600, random_state=random_state),
            'Random Forest': RandomForestNumPy(n_trees=20, max_depth=6, random_state=random_state),
            'XGBoost / GBDT': GradientBoostingNumPy(n_estimators=20, learning_rate=0.1, max_depth=4, random_state=random_state)
        }
        self.trained_wrappers = {}
        self.evaluation_results = {}
        self.best_model_name = None
        self.best_wrapper = None

    def train_all(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        best_auc = -1.0
        
        for name, clf in self.models.items():
            wrapper = CreditRiskPipelineWrapper(name, clf)
            wrapper.fit(X_train, y_train)
            self.trained_wrappers[name] = wrapper
            
            y_pred = wrapper.predict(X_test)
            y_proba = wrapper.predict_proba(X_test)[:, 1]
            y_test_vals = y_test.values
            
            auc = roc_auc_score_numpy(y_test_vals, y_proba)
            gini = calculate_gini_coefficient(auc)
            ks = calculate_ks_statistic(y_test_vals, y_proba)
            acc = float(np.mean(y_pred == y_test_vals))
            
            tp = np.sum((y_pred == 1) & (y_test_vals == 1))
            fp = np.sum((y_pred == 1) & (y_test_vals == 0))
            fn = np.sum((y_pred == 0) & (y_test_vals == 1))
            tn = np.sum((y_pred == 0) & (y_test_vals == 0))
            
            prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            
            results = {
                'accuracy': round(acc, 4),
                'roc_auc': round(auc, 4),
                'gini': round(gini, 4),
                'ks_statistic': round(ks, 4),
                'precision': round(prec, 4),
                'recall': round(rec, 4),
                'f1_score': round(f1, 4),
                'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]]
            }
            
            self.evaluation_results[name] = results
            
            if auc > best_auc:
                best_auc = auc
                self.best_model_name = name
                self.best_wrapper = wrapper
                
        return {
            'best_model_name': self.best_model_name,
            'evaluations': self.evaluation_results
        }
