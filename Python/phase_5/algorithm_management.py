# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy"
# ]
# ///

import marimo

__generated_with = "0.10.16"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import time
    return mo, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # 🧩 Algorithm Development Mastery

        Welcome to the algorithmic playground where code becomes an art form!

        ## The Algorithm Architect's Toolkit

        Algorithms are like recipes for solving computational problems. 
        Just as a chef breaks down a complex dish into simple steps, 
        we'll learn to decompose problems and create efficient solutions.

        """
    )
    return


@app.cell
def _(problem_types):
    problem_types
    return


@app.cell
def _(mo):
    # Problem Decomposition Demonstrator
    problem_types = mo.ui.dropdown(
        options=[
            "Finding Maximum in List", 
            "Sorting Numbers", 
            "String Manipulation", 
            "Mathematical Calculations"
        ],
        label="Select a Problem Type"
    )
    return (problem_types,)


@app.cell(hide_code=True)
def _(mo, problem_types):
    # Problem Decomposition Explanations
    decomposition_examples = {
        "Finding Maximum in List": mo.md("""
        ### 🏆 Maximum Value Problem Decomposition

        **Naive Approach:**
        ```python
        def find_max(numbers):
            max_value = numbers[0]
            for num in numbers:
                if num > max_value:
                    max_value = num
            return max_value
        ```

        **Decomposition Steps:**
        
        1. Initialize with first element
        
        2. Compare each element
        
        3. Update max if larger found
        
        4. Return maximum value

        **Optimization Potential:**
        
        - Use built-in `max()` function
        
        - Implement more efficient comparison strategies
        """),

        "Sorting Numbers": mo.md("""
        ### 🔢 Sorting Problem Decomposition

        **Naive Approach (Bubble Sort):**
        ```python
        def bubble_sort(arr):
            n = len(arr)
            for i in range(n):
                for j in range(0, n-i-1):
                    if arr[j] > arr[j+1]:
                        arr[j], arr[j+1] = arr[j+1], arr[j]
            return arr
        ```

        **Decomposition Steps:**

        1. Iterate through the list
        
        2. Compare adjacent elements
        
        3. Swap if in wrong order
        
        4. Repeat until sorted

        **Optimization Potential:**
        
        - Use built-in `sorted()` function
        
        - Implement more efficient algorithms like QuickSort
        """),

        "String Manipulation": mo.md("""
        ### 📝 String Processing Problem Decomposition

        **Problem: Count Word Frequencies**
        ```python
        def word_frequency(sentence):
            # Split into words
            words = sentence.lower().split()

            # Count frequencies
            freq_dict = {}
            for word in words:
                if word in freq_dict:
                    freq_dict[word] += 1
                else:
                    freq_dict[word] = 1

            return freq_dict
        ```

        **Decomposition Steps:**
        
        1. Normalize text (lowercase)
        
        2. Split into individual words
        
        3. Create frequency dictionary
        
        4. Count occurrences

        **Optimization Potential:**
        
        - Use `collections.Counter()`
        
        - Implement more efficient counting methods
        """),

        "Mathematical Calculations": mo.md("""
        ### 🧮 Mathematical Problem Decomposition

        **Problem: Fibonacci Sequence**
        ```python
        def fibonacci(n):
            # Recursive (inefficient) approach
            if n <= 1:
                return n
            return fibonacci(n-1) + fibonacci(n-2)
        ```

        **Decomposition Steps:**
        
        1. Define base cases
        
        2. Break problem into smaller subproblems
        
        3. Combine results
        
        4. Avoid redundant calculations

        **Optimization Potential:**
        
        - Use dynamic programming
        
        - Implement memoization
        
        - Utilize iterative approaches
        """)
    }

    decomposition_examples.get(problem_types.value, mo.md("Select a problem type"))
    return (decomposition_examples,)


@app.cell
def _(algorithm_selector, input_n, mo):
    mo.hstack([algorithm_selector, input_n])
    return


@app.cell
def _(mo):
    # Efficiency Comparison Playground
    algorithm_selector = mo.ui.dropdown(
        options=[
            "Fibonacci (Recursive)", 
            "Fibonacci (Dynamic Programming)", 
            "Fibonacci (Iterative)"
        ],
        label="Select Fibonacci Implementation"
    )
    input_n = mo.ui.number(
        value=30, 
        label="Calculate nth Fibonacci Number", 
        start=0, 
        stop=40
    )
    return algorithm_selector, input_n


@app.cell
def _(performance_display):
    performance_display
    return


@app.cell
def _(algorithm_selector, input_n, mo, time):
    # Fibonacci Implementations with Performance Tracking
    def fibonacci_recursive(n):
        if n <= 1:
            return n
        return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)

    def fibonacci_dynamic(n):
        # Dynamic Programming approach
        if n <= 1:
            return n

        # Initialize memoization array
        fib = [0] * (n + 1)
        fib[1] = 1

        # Build solution bottom-up
        for i in range(2, n + 1):
            fib[i] = fib[i-1] + fib[i-2]

        return fib[n]

    def fibonacci_iterative(n):
        # Iterative approach with minimal memory
        if n <= 1:
            return n

        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b

        return b

    # Performance measurement
    def measure_performance(func, n):
        start_time = time.time()
        result = func(n)
        end_time = time.time()
        return result, (end_time - start_time) * 1000  # Convert to milliseconds

    # Select algorithm based on dropdown
    algorithm_map = {
        "Fibonacci (Recursive)": fibonacci_recursive,
        "Fibonacci (Dynamic Programming)": fibonacci_dynamic,
        "Fibonacci (Iterative)": fibonacci_iterative
    }

    selected_algorithm = algorithm_map[algorithm_selector.value]

    # Measure performance
    result, execution_time = measure_performance(selected_algorithm, input_n.value)

    # Display results
    performance_display = mo.hstack([
        mo.md(f"""
        ### 🧮 Fibonacci Performance

        **Algorithm**: {algorithm_selector.value}

        **Input**: n = {input_n.value}

        **Result**: {result}

        **Execution Time**: {execution_time:.4f} ms
        """),
        mo.md(f"""
        ### 📊 Efficiency Analysis

        **Performance Characteristics**:
        
        - Time Complexity
        
        - Space Complexity
        
        - Scalability

        **Rank**: {'⭐' * (3 - list(algorithm_map.keys()).index(algorithm_selector.value))}
        """)
    ])
    return (
        algorithm_map,
        execution_time,
        fibonacci_dynamic,
        fibonacci_iterative,
        fibonacci_recursive,
        measure_performance,
        performance_display,
        result,
        selected_algorithm,
    )


@app.cell
def _(complexity_selector):
    complexity_selector
    return


@app.cell(hide_code=True)
def _(mo):
    # Big O Notation Explorer
    complexity_selector = mo.ui.dropdown(
        options=[
            "O(1) - Constant Time", 
            "O(log n) - Logarithmic", 
            "O(n) - Linear", 
            "O(n log n) - Linearithmic", 
            "O(n²) - Quadratic"
        ],
        label="Select Time Complexity"
    )
    return (complexity_selector,)


@app.cell(hide_code=True)
def _(complexity_selector, mo):
    # Big O Notation Explanations
    big_o_explanations = {
        "O(1) - Constant Time": mo.md("""
        ### 🚀 O(1) - Constant Time Complexity

        **Characteristics:**
        
        - Performance remains constant
        
        - Execution time doesn't change with input size

        **Examples:**
        ```python
        def get_first_element(arr):
            return arr[0]  # Always takes same time

        def check_if_even(num):
            return num % 2 == 0  # Single operation
        ```

        **Real-world Analogy:**
        Like grabbing the first item from a shelf, 
        no matter how many items are on the shelf.
        """),

        "O(log n) - Logarithmic": mo.md("""
        ### 🌳 O(log n) - Logarithmic Time Complexity

        **Characteristics:**
        
        - Performance grows slowly
        
        - Typically seen in divide-and-conquer algorithms

        **Examples:**
        ```python
        def binary_search(arr, target):
            left, right = 0, len(arr) - 1
            while left <= right:
                mid = (left + right) // 2
                if arr[mid] == target:
                    return mid
                elif arr[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            return -1
        ```

        **Real-world Analogy:**
        Like finding a name in a phone book by 
        eliminating half the pages each time.
        """),

        "O(n) - Linear": mo.md("""
        ### 📏 O(n) - Linear Time Complexity

        **Characteristics:**
        
        - Performance grows linearly with input size
        
        - Single pass through all elements

        **Examples:**
        ```python
        def find_max(arr):
            max_val = arr[0]
            for num in arr:
                if num > max_val:
                    max_val = num
            return max_val

        def linear_search(arr, target):
            for i, val in enumerate(arr):
                if val == target:
                    return i
            return -1
        ```

        **Real-world Analogy:**
        Like counting items in a shopping cart - 
        more items mean more time.
        """),

        "O(n log n) - Linearithmic": mo.md("""
        ### 🔄 O(n log n) - Linearithmic Time Complexity

        **Characteristics:**
        
        - Efficient for sorting and searching
        
        - Balances between linear and quadratic

        **Examples:**
        ```python
        def merge_sort(arr):
            if len(arr) <= 1:
                return arr

            mid = len(arr) // 2
            left = merge_sort(arr[:mid])
            right = merge_sort(arr[mid:])

            return merge(left, right)

        def merge(left, right):
            result = []
            i = j = 0
            while i < len(left) and j < len(right):
                if left[i] < right[j]:
                    result.append(left[i])
                    i += 1
                else:
                    result.append(right[j])
                    j += 1
            result.extend(left[i:])
            result.extend(right[j:])
            return result
        ```

        **Real-world Analogy:**
        Like efficiently organizing a library 
        by systematically sorting books.
        """),

        "O(n²) - Quadratic": mo.md("""
        ### 🔢 O(n²) - Quadratic Time Complexity

        **Characteristics:**
        - Performance degrades quickly with input size
        
        
        - Nested loops are typical

        **Examples:**
        ```python
        def bubble_sort(arr):
            n = len(arr)
            for i in range(n):
                for j in range(0, n-i-1):
                    if arr[j] > arr[j+1]:
                        arr[j], arr[j+1] = arr[j+1], arr[j]
            return arr

        def find_duplicates(arr):
            duplicates = []
            for i in range(len(arr)):
                for j in range(i+1, len(arr)):
                    if arr[i] == arr[j]:
                        duplicates.append(arr[i])
            return duplicates
        ```

        **Real-world Analogy:**
        Like comparing every student with every 
        other student in a classroom.
        """)}

    big_o_explanations.get(complexity_selector.value, mo.md("Select a complexity"))
    return (big_o_explanations,)


@app.cell(hide_code=True)
def _(mo):
    callout_text = mo.md("""
    ## Your Algorithm Mastery Journey!

    Next Steps:

    - Practice problem decomposition
    - Implement different algorithm strategies
    - Analyze time and space complexity

    You're becoming an Algorithm Wizard! 🧙‍♂️🔍
    """)

    mo.callout(callout_text, kind="success")
    return (callout_text,)


if __name__ == "__main__":
    app.run()
