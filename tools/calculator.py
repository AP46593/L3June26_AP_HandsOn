"""
calculator.py - Simple calculator tool.
Operations: add, subtract, multiply, divide, square, cube, sqrt, cbrt.
Returns formatted result string or error message for unsupported operations.
"""

import math
from langchain_core.tools import tool


@tool
def simple_calc(operation: str, a: float, b: float = 0) -> str:
    """Perform a simple calculation.

    Supported operations:
      - add: a + b
      - subtract: a - b
      - multiply: a * b
      - divide: a / b
      - square: a^2
      - cube: a^3
      - sqrt: square root of a
      - cbrt: cube root of a

    Args:
        operation: One of add, subtract, multiply, divide, square, cube, sqrt, cbrt.
        a: First number.
        b: Second number (used for add, subtract, multiply, divide).
    """
    op = operation.lower().strip()

    if op == "add":
        result = a + b
    elif op == "subtract":
        result = a - b
    elif op == "multiply":
        result = a * b
    elif op == "divide":
        if b == 0:
            return "Error: Division by zero."
        result = a / b
    elif op == "square":
        result = a ** 2
    elif op == "cube":
        result = a ** 3
    elif op == "sqrt":
        if a < 0:
            return "Error: Cannot take square root of a negative number."
        result = math.sqrt(a)
    elif op == "cbrt":
        result = round(a ** (1 / 3), 10)
    else:
        return f"Unsupported operation: {operation}. Supported: add, subtract, multiply, divide, square, cube, sqrt, cbrt."

    return f"{operation}({a}, {b}) = {result}" if op in ("add", "subtract", "multiply", "divide") else f"{operation}({a}) = {result}"
