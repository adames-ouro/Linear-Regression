### Linear Regression

- A basic linear regression is a statistical model that represent the relationship between predictor variables ***X***<sub>i</sub> with a response variable ***Y***. 

- The model assumes a linear relation where the intercept of the line is usually represented by  $\beta_{0}$ and the slope of the variables by a  coefficient $\beta_{n}$ . 

$$ Y = \beta_{0} + \beta_{1} * X_{1} + ... +\beta_{n} * X_{n} + \varepsilon $$

- The equation can be written in linear algebra notation if a columns of 1 is added to the ***X*** matrix:

$$ Y =  \beta . X + \varepsilon $$

- The predicted values are often called fitted values and symbolized as $\hat{Y}$.

$$ \hat{Y} =  \hat{\beta} * X $$

- The coefficients $\hat{\beta_{i}}$ can be calculated as:

 $$\hat{{\beta}} = Inverse((X^T . X)). X^T . Y $$

- The intercept is in the same position as the column of 1s in the ***X*** matrix, the rest of the values are slopes for ***Xi***


## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install foobar
```

## Usage

```python
import foobar

# returns 'words'
foobar.pluralize('word')

# returns 'geese'
foobar.pluralize('goose')

# returns 'phenomenon'
foobar.singularize('phenomena')
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
