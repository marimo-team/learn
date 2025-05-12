import marimo

__generated_with = "0.12.9"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""**Notebook 1: Introduction to NumPy**""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        **What is NumPy?**

        NumPy stands for *numerical python* and is one of the foundational libraries for numerical computing and operations in Python! It's efficient and powerful, and introduces users to working with datasets, simulations, image processing, and more! In fact, NumPy introduced the **array**, a fundamental data structure in Python.

        Let's begin. In your terminal, make sure to activate your environment and install NumPy. You can do this by typing:

        *pip install NumPy*

        **Great! In this notebook, we'll:**

        - Learn what NumPy is and why it's widely used.
        - See some cool demos showing off what NumPy can do.
        - Run real-world examples.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    # Let's start by importing numpy.

    import numpy as np
    return (np,)


@app.cell
def _(mo):
    mo.md(
        r"""
        **Simulate 1 Million Dice Rolls**

        Let's start with a simple but powerful example of NumPy. Using the function *np.random.randint*, we can simulate the roll of a die one million times. In regular Python, we'd probably need a huge loop - but with NumPy, we can do it in one line. 
        """
    )
    return


@app.cell
def _(np):
    rolls = np.random.randint(1, 7, size = 1000000)
    return (rolls,)


@app.cell
def _(mo):
    mo.md(
        r"""
        **What's happening here?**

        The code above creates *one million random integers* between 1 and 6 (inclusive), simulating the rolling of a regular 6-sided die a million times. **Rolls** is now a NumPy array holding all those results!
        """
    )
    return


@app.cell
def _(np, rolls):
    # Let's analyze the dice rolls. 

    print("Average roll: ", np.mean(rolls))
    print("Standard deviation: ", np.std(rolls))
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        That was a simple example. Let's look at something a little more in depth, using another library, **Matplotlib**. 

        **Gradient Heatmap**

        Here, we'll create a colorful 2D graph of a math function using NumPy and Matplotlib!
        """
    )
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    return (plt,)


@app.cell
def _(np):
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)

    X, Y = np.meshgrid(x, y)
    Z = np.sin(X ** 2 + Y ** 2)
    return X, Y, Z, x, y


@app.cell
def _(Z, plt):
    plt.imshow(Z, cmap='Accent', extent=(-3, 3, -3, 3))
    plt.title("Sinusoidal Gradient")
    plt.colorbar()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        **What's going on here?**

        Well, we used *np.linspace()*, which allowed us to create 100 values between -3 and 3. The function *np.meshgrid()* allowed us to turn x and y into 2D coordinates and plot them on a graph. Lastly, *np.sin()* applied a math function to every grid point. *Plt.imshow()*, from the Matplotlib library, allows us to visualize it. 

        It's understanding if this is confusing! But this just shows us how NumPy can be used in combination with other libraries to create pretty cool visualizations.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        **Simulating Human Height Data**

        We can also use numpy to analyze data. For example, let's pretend we're studying a sample of 10,000 people and we want to analyze their heights.
        """
    )
    return


@app.cell
def _(np):
    heights = np.random.normal(loc = 170, scale = 10, size = 10000)
    return (heights,)


@app.cell
def _(mo):
    mo.md(
        r"""
        **What's happening here?**

        *np.random.normal()* generates for us random values from a normal distribution. We pass in certain information, like *loc = 170* (The average height is 170 cm), *scale = 10* (The standard deviation is 10 cm) and *size = 10000* (We are simulating 10,000 individuals).
        """
    )
    return


@app.cell
def _(heights, plt):
    # Just like in the last example, let's visualize the data. 

    plt.hist(heights, bins=50, color='skyblue', edgecolor='black')
    plt.title("Simulated Human Heights")
    plt.xlabel("Height (cm)")
    plt.ylabel("Number of People")
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""A bell curve - normally distributed, and most people are near the average.""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        **Predicting Loan Payments**

        Let’s say you want to model how much someone would pay per month for a small loan — similar to what a banker or budgeting app might do. This example shows us how NumPy can do financial math!
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        We'll use the basic loan formula:

        $$
        \text{Monthly Payment} = \frac{P \cdot r}{1 - (1 + r)^{-n}}
        $$

        Where P is the loan amount, r is the monthly interest rate, and n is the number of monthly payments.
        """
    )
    return


@app.cell
def monthly_payment():
    # First, let's create a function to define the loan formula. 

    def monthly_payment(principal, annual_rate, months):
        r = annual_rate / 12  # Convert to monthly interest rate
        payment = (principal * r) / (1 - (1 + r) ** -months)
        return payment

    return (monthly_payment,)


@app.cell
def _(monthly_payment):
    # Now we can experiment!

    loan = 10000
    rate = 0.05
    months = 36

    payment = monthly_payment(loan, rate, months)
    print(f"Monthly payment: ${payment: .2f}")
    return loan, months, payment, rate


@app.cell
def _(mo):
    mo.md(r"""Seems pretty simple - now let's see if we can calculate payments for many loan amounts at once.""")
    return


@app.cell
def _(loan, monthly_payment, np):
    loan_amounts = np.arange(1_000, 11_000, 1_000)
    new_rate = 0.05
    new_months = 36

    new_payments = monthly_payment(loan_amounts, new_rate, new_months)

    for loan_item, pmt in zip(loan_amounts, new_payments):
        print(f"Loan: ${loan}, Monthly Payment: ${pmt:.2f}")
    return loan_amounts, loan_item, new_months, new_payments, new_rate, pmt


@app.cell
def _(mo):
    mo.md(r"""Because loan_ammounts is a NumPy array, the function calculates all our payments at once! Let's take a look at how to plot it.""")
    return


@app.cell
def _(loan_amounts, new_payments, plt):
    plt.plot(loan_amounts, new_payments, marker='o')
    plt.title("Loan Amount vs. Monthly Payment")
    plt.xlabel("Loan Amount")
    plt.ylabel("Monthly Payment")
    plt.grid(True)
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""The graph shows us a straight line - as the loan increases, so does the payment. With NumPy, we did all the math with just one function!""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        **Bonus Challenge**

        Can you calculate how much interest you would pay in total over the entire loan?
        """
    )
    return


@app.cell
def _():
    # Your code here!
    # Hint: What function can you use to add everything together?
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        **Notebook Summary**

        In this notebook, we explored some of the exciting things NumPy can do — even if you're new to coding.
        """
    )
    return


if __name__ == "__main__":
    app.run()
