# Linear Regression

---

### Why Linear Regression?
- Simple and interpretable model for understanding relationships between features and target variables.
- Requires relatively less computational power.
- Provides a foundation for many advanced modeling techniques.
- Can be extended to handle non-linear relationships and interactions.

---

## Ordinary Least Squares (OLS)

---

### Goal
To find the best-fitting line that minimizes the sum of squared differences (residuals) between observed and predicted values.

### Overview
- Supervised learning method used for both continuous and categorical variables.
- Assumes a linear relationship between dependent and independent variables.
- Involves Hyperparameters when regularized (like in Ridge or Lasso regression).
- Sensitive to outliers.
- Requires assumption checks: linearity, independence, homoscedasticity, and normality.

### Steps for the algorithm

**Step 1: Model Representation**  
Choose the form of the function.

**Step 2: Define the Objective Function**  
Sum of squared differences between observed and predicted values (RSS - Residual Sum of Squares).

**Step 3: Minimize the Objective Function**  
Use calculus (for simple linear regression) or numerical methods (for multiple regression) to find the \(\beta\) values that minimize RSS.

**Step 4: Model Evaluation**  
Use metrics such as R-squared, RMSE, etc. to evaluate the fit of the model.

**Step 5: Prediction**  
Use the fitted model to make predictions on new data.

> **Note:** While Linear Regression is powerful and simple, it's important to check the underlying assumptions and understand its limitations when analyzing data.


### Statistical Model

- A basic linear regression is a statistical model that represent the relationship between predictor variables ***X***<sub>i</sub> with a response variable ***Y***. 

- The model assumes a linear relation where the intercept of the line is usually represented by  $\beta_{0}$ and the slope of the variables by a  coefficient $\beta_{n}$ . 

$$ Y = \beta_{0} + \beta_{1} * X_{1} + ... +\beta_{n} * X_{n} + \varepsilon $$

- The equation can be written in linear algebra notation if a columns of 1 is added to the ***X*** matrix:

$$ Y =  \beta . X + \varepsilon $$

- The predicted values are often called fitted values and symbolized as $\hat{Y}$.

$$ \hat{Y} =  \hat{\beta} * X $$

- The coefficients $\hat{\beta_{i}}$ can be calculated as:

 $$\hat{{\beta}} = Inverse((X^T . X)). X^T . Y $$

- The intercept is in the same position as the column of 1s in the ***X*** matrix, the rest of the values are slopes for ***X***<sub>i</sub>



# Linear Regression with PyTorch

---

### Why Linear Regression in PyTorch?
- PyTorch provides an efficient and easy way to define, train, and evaluate models with automatic differentiation.
- Easily scalable to more complex models and architectures.
- Can utilize GPU acceleration for faster computation.

---

## Neural Network Representation

---

### Goal
To leverage PyTorch's neural network framework to implement and train a linear regression model.

### Overview
- A single-layer neural network without activation can represent a linear regression model.
- Loss function used is Mean Squared Error (MSE) for regression tasks.
- Gradient descent or its variants (like Adam, SGD) can be used for optimization.

### Steps for the algorithm in PyTorch

**Step 1: Define the Model**  
A simple linear layer with input and output size set to the number of features and 1 respectively.

**Step 2: Define the Loss Function and Optimizer**  
Typically, Mean Squared Error (MSE) is used for regression problems.

**Step 3: Model Training**  
Iterate through data, forward pass, compute loss, backward pass, and update weights.

**Step 4: Model Evaluation**
Use the trained model to make predictions and evaluate them against known targets.

**Step 5: Model Deployment**
With a trained model, you can save and load the model for future predictions on new data.

> **Note:** While Linear Regression is powerful and simple, it's important to check the underlying assumptions and understand its limitations when analyzing data.

# Neural Networks with PyTorch

---

## 1. Basics of Neural Networks

Neural Networks consist of layers of interconnected nodes (or neurons). They can be seen as a series of matrix operations, with non-linear activation functions introduced between these operations.

A simple feed-forward neural network can be represented as:

$$
\mathbf{a}^{[l]} = \mathbf{z}^{[l]} + \mathbf{b}^{[l]}
$$

$$
\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]}
$$

$$
\mathbf{a}^{[l]} = g^{[l]}(\mathbf{z}^{[l]})
$$

Where:
- $\mathbf{W}^{[l]}$ is the weight matrix for layer \( l \)

- $\mathbf{b}^{[l]}$ is the bias vector for layer \( l \)

- $\mathbf{a}^{[l]}$ is the activation of layer \( l \)

- $g^{[l]}$ is the activation function for layer \( l \)

---

## 2. Activation Functions

PyTorch supports a variety of activation functions, some of which include:

1. **Sigmoid**:
   
    $ \sigma(z) = \frac{1}{1 + e^{-z}} $

2. **Tanh**:
   
    $ \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} $

3. **ReLU**:
   
    $ \text{ReLU}(z) = \max(0, z) $


---

## 3. Loss Functions

Depending on the task (regression, classification, etc.), different loss functions can be used:

1. **Mean Squared Error (for regression)**:
    $$ \text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2 $$

2. **Cross Entropy Loss (for classification)**:
    $$ \text{CrossEntropy}(y, \hat{y}) = -\sum_{i} y_i \log(\hat{y}_i) $$

---

## 4. Optimization

Gradient descent and its variants are used to optimize the weights of the network:

1. **Gradient Descent**:
    $$ \mathbf{W} = \mathbf{W} - \alpha \nabla_{\mathbf{W}} J(\mathbf{W}, \mathbf{b}) $$

2. **Stochastic Gradient Descent (SGD)**:
    Update weights after each training example.

3. **Momentum, RMSProp, Adam**:
    More advanced optimization methods that combine various techniques for faster convergence.

> **Note**: PyTorch provides automatic differentiation, which means you don't have to manually compute the gradient. Instead, you can use PyTorch's autograd functionality to compute the backward pass.

---

## 5. Backpropagation

The key to training a neural network is the backpropagation algorithm, which computes the gradient of the loss with respect to each weight by applying the chain rule.

Given the forward propagation equations above, the gradients can be calculated using the chain rule.

---

**Remember:** While PyTorch handles much of the intricacies of backpropagation and optimization, understanding the foundational math helps in building, debugging, and optimizing neural network models.


