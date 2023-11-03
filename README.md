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



## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
