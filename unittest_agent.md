# Unit Test Agent

## Agent Configuration

**Name:** Unit Test Generator Agent  
**Purpose:** Automatically generate comprehensive unit tests for code files  
**Model:** Claude Sonnet 4.5

## Agent Capabilities

1. **Test Generation**
   - Generate unit tests for Python, JavaScript, TypeScript, Java, C#, and more
   - Create test cases covering edge cases, normal cases, and error handling
   - Follow testing best practices and conventions

2. **Test Frameworks**
   - Python: pytest, unittest
   - JavaScript/TypeScript: Jest, Mocha, Jasmine
   - Java: JUnit
   - C#: NUnit, xUnit

3. **Coverage Analysis**
   - Identify untested code paths
   - Suggest additional test cases for better coverage
   - Generate mock objects and test fixtures

## Agent Instructions

When generating unit tests, the agent will:

1. **Analyze the code** to understand:
   - Function signatures and parameters
   - Expected inputs and outputs
   - Error conditions and edge cases
   - Dependencies and external calls

2. **Generate tests** that include:
   - Setup and teardown methods
   - Positive test cases (happy path)
   - Negative test cases (error handling)
   - Edge cases (boundary conditions, null values, empty inputs)
   - Mock external dependencies
   - Assertions for expected behavior

3. **Follow naming conventions**:
   - Descriptive test names (e.g., `test_login_with_valid_credentials`)
   - Arrange-Act-Assert pattern
   - Clear test documentation

4. **Provide coverage recommendations**:
   - Suggest missing test scenarios
   - Identify untested code branches
   - Recommend integration tests where needed

## Usage Examples

### Example 1: Generate tests for a Python function
```
@agent Generate unit tests for the calculate_discount function using pytest
```

### Example 2: Generate tests for a class
```
@agent Create comprehensive unit tests for the UserService class with mocking
```

### Example 3: Add missing test cases
```
@agent Analyze coverage and suggest additional test cases for this module
```

## Test Template Structure

### Python (pytest)
```python
import pytest
from module import function_name

class TestFunctionName:
    def setup_method(self):
        # Setup code
        pass
    
    def test_normal_case(self):
        # Arrange
        # Act
        # Assert
        pass
    
    def test_edge_case(self):
        pass
    
    def test_error_handling(self):
        with pytest.raises(ExpectedException):
            pass
```

### JavaScript (Jest)
```javascript
describe('FunctionName', () => {
    beforeEach(() => {
        // Setup
    });
    
    test('should handle normal case', () => {
        // Arrange
        // Act
        // Assert
        expect().toBe();
    });
    
    test('should handle edge case', () => {
        // Test implementation
    });
    
    test('should throw error on invalid input', () => {
        expect(() => {}).toThrow();
    });
});
```

## Best Practices

1. **Independence**: Each test should run independently
2. **Clarity**: Test names should describe what is being tested
3. **Simplicity**: One assertion per test when possible
4. **Speed**: Tests should run quickly
5. **Reliability**: Tests should be deterministic
6. **Maintainability**: Keep tests simple and easy to update

## Code Coverage Goals

- **Minimum**: 80% code coverage
- **Target**: 90%+ code coverage
- **Critical paths**: 100% coverage for business logic

## Agent Triggers

Use these commands to invoke the unit test agent:

- `Generate unit tests for [file/function/class]`
- `Create test cases for [component]`
- `Add missing test coverage for [module]`
- `Review and improve existing tests`
- `Generate mock objects for [dependency]`

## Output Format

The agent will provide:
1. Complete test file with all necessary imports
2. Test class/suite organization
3. Individual test cases with comments
4. Setup and teardown methods
5. Mock configurations if needed
6. Coverage recommendations