# Contributing to Earth Sciences Python Library

Thank you for considering contributing! This is an alpha-stage project that needs help from the community.

## Getting Started

### Development Setup

1. **Clone the repository**
   ```bash
   cd "/Users/k.jones/Desktop/earth science"
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Run tests to verify setup**
   ```bash
   pytest tests/ -v
   ```

## How to Contribute

### Priority Areas

We especially need help with:

1. **Validation** - Verify outputs against known results and published data
2. **Testing** - Add tests, especially for edge cases
3. **Performance** - Optimize slow functions (especially kriging)
4. **Documentation** - Improve docstrings, add examples
5. **Bug Fixes** - Fix issues listed in KNOWN_ISSUES.md
6. **Input Validation** - Add proper input checking

### Making Changes

1. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```

2. **Make your changes**
   - Follow existing code style
   - Add/update tests
   - Update documentation
   - Add validation results (if applicable)

3. **Run tests and linting**
   ```bash
   # Run tests
   pytest tests/ -v
   
   # Run with coverage
   pytest tests/ --cov=earthsciences --cov-report=html
   
   # Format code
   black earthsciences/ tests/
   
   # Check imports
   isort earthsciences/ tests/
   
   # Lint
   flake8 earthsciences/ tests/
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "Brief description of changes"
   ```

5. **Push and create pull request**
   ```bash
   git push origin feature/your-feature-name
   ```

## Code Style

### Python Style
- Follow PEP 8
- Use Black for formatting (line length: 100)
- Use type hints where possible
- Write clear docstrings (numpy style)

### Docstring Format
```python
def my_function(param1: np.ndarray, param2: float = 1.0) -> Dict:
    """
    Brief one-line description.
    
    Longer description if needed. Explain what the function does,
    any important algorithms, and relevant references.
    
    Parameters
    ----------
    param1 : array_like
        Description of param1
    param2 : float, optional
        Description of param2 (default=1.0)
        
    Returns
    -------
    dict
        Dictionary containing:
        - key1: description
        - key2: description
        
    Raises
    ------
    ValueError
        When invalid input is provided
        
    Examples
    --------
    >>> result = my_function(np.array([1, 2, 3]))
    >>> print(result['key1'])
    
    Notes
    -----
    Any important notes about the implementation, limitations,
    or references to papers/textbooks.
    
    References
    ----------
    .. [1] Author et al. (Year). "Title". Journal.
    """
    # Implementation
    pass
```

### Testing

All new functions must have tests:

```python
class TestMyFunction:
    """Test my_function."""
    
    def test_basic_functionality(self):
        """Test that my_function works with valid input."""
        result = my_function(np.array([1, 2, 3]))
        assert 'key1' in result
    
    def test_edge_case(self):
        """Test edge cases."""
        result = my_function(np.array([]))
        # assertions
    
    def test_invalid_input(self):
        """Test that invalid input raises appropriate error."""
        with pytest.raises(ValueError):
            my_function(np.array([1]), param2=-1)
```

## Validation Guidelines

When modifying or adding statistical/scientific functions:

### 1. Compare with Known Results

```python
# Use published datasets or standard test cases
import numpy as np
from earthsciences import ...

data = np.array([1, 2, 3, 4, 5])
result = some_function(data)

# Compare with known results from papers or textbooks
```

### 2. Use Standard Test Cases

- Standard datasets from scientific literature
- Synthetic data with known properties
- Examples from published papers

### 3. Add Validation Tests

```python
def test_validates_against_known_result():
    """Validate against published example."""
    # Data from paper/textbook
    data = np.array([...])
    
    # Expected output
    expected_mean = 5.234
    expected_std = 1.456
    
    # Test
    result = descriptive_stats(data)
    assert np.isclose(result['mean'], expected_mean, atol=1e-3)
    assert np.isclose(result['std'], expected_std, atol=1e-3)
```

## Performance Contributions

If optimizing performance:

1. **Benchmark before and after**
   ```python
   import time
   
   # Before
   start = time.time()
   result = slow_function(large_data)
   print(f"Old: {time.time() - start:.2f}s")
   
   # After
   start = time.time()
   result = fast_function(large_data)
   print(f"New: {time.time() - start:.2f}s")
   ```

2. **Ensure results don't change** (or document changes)

3. **Add performance test** (marked as slow)
   ```python
   @pytest.mark.slow
   def test_performance():
       """Benchmark function performance."""
       # Test with large dataset
       # Add assertion on max time
   ```

## Documentation Contributions

### Improving Docstrings
- Add examples
- Clarify parameters
- Add references to papers/textbooks
- Note limitations

### Adding Tutorials
- Jupyter notebooks in `examples/`
- Focus on real earth science applications
- Include data sources
- Explain methodology

### API Documentation
- Build with Sphinx (when configured)
- Keep up to date with code

## Bug Reports

Good bug reports include:

1. **Minimal reproducible example**
   ```python
   import numpy as np
   from earthsciences import ...
   
   data = np.array([1, 2, 3])  # Minimal data
   result = function(data)      # Call that fails
   # Error occurs here
   ```

2. **Expected vs actual behavior**
   - What you expected
   - What actually happened
   - Reference implementation output (if available)

3. **Environment**
   - Python version
   - Library version
   - OS
   - Key dependency versions

## Questions?

- Open an issue for questions
- Check KNOWN_ISSUES.md
- Review existing code for patterns

## Code of Conduct

- Be respectful and constructive
- Focus on improving the library
- Help others learn
- Credit sources appropriately

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
