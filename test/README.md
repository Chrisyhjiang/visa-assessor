# Test Scripts

This directory contains test scripts for the O-1A Visa Qualification Assessment System.

## Files

- **test_system.py**: Tests core functionality of the system
- **test_api.py**: Tests the API with mock LLM responses
- **test_client.py**: Tests the API by sending a sample CV
- **run_tests.py**: Script to run all tests

## Usage

### Running All Tests

```bash
python test/run_tests.py
```

### Running Individual Tests

```bash
python test/test_system.py
python test/test_api.py
python test/test_client.py
```
