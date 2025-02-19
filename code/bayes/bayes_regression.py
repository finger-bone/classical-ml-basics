import numpy as np

class BayesLinearRegression:
    def __init__(self, tau=1.0, sigma=1.0):
        self.tau = tau
        self.sigma = sigma
        self.beta_post = None
        self.sigma_post = None

    def fit(self, X, y):
        # Prior covariance (tau^2 I)
        tau2_I = self.tau ** 2 * np.eye(X.shape[1])

        # Likelihood covariance (sigma^2 I)
        sigma2_I = self.sigma ** 2 * np.eye(X.shape[0])

        # Compute the posterior covariance
        XTX = X.T @ X
        self.sigma_post = np.linalg.inv(np.linalg.inv(tau2_I) + (1 / self.sigma**2) * XTX)

        # Compute the posterior mean
        self.beta_post = (1 / self.sigma**2) * self.sigma_post @ X.T @ y

    def predict(self, X_new):
        # Predictive mean
        y_pred_mean = X_new @ self.beta_post
        
        # Predictive covariance
        y_pred_cov = self.sigma ** 2 + np.sum(X_new @ self.sigma_post * X_new, axis=1)
        
        return y_pred_mean, y_pred_cov

x = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [7, 8]])
y = np.array([3, 5, 7, 9, 11, 15])
b_lr = BayesLinearRegression(tau=1.0, sigma=1.0)
b_lr.fit(x, y)
y_pred_mean, y_pred_cov = b_lr.predict(np.array([[0, 1]]))
print(y_pred_mean, y_pred_cov)
