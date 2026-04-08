marimo for Educators

## Introduction

A computational notebook is a form of *literate programming* that mixes prose and software in a single runnable document. Each *cell* contains either prose written in Markdown or code written in a supported programming language, and the output of any code cell is displayed directly in the notebook alongside the source.

Notebooks make everyday work easier to understand because they present explanation and evidence together, just like people do in conversation. This format also improves reproducibility, since the code that produced a result lives beside the result itself.

For learning, notebooks offer several advantages over static material. Learners become active participants rather than passive readers: they can experiment with settings, alter code, and see immediate results. Because notebooks can be hosted, learners do not need to install anything before they start, which means the first session can focus on the subject itself rather than on tooling. The format is less intimidating than jumping straight into a script editor, and it introduces learners to a real-world tool used in research and industry.

For teaching, notebooks serve all those purposes while also letting instructors keep interactive lecture material in one place.

### Why marimo?

marimo is open source and occupies a useful middle ground: richer than a plain text editor but less overwhelming than a full IDE such as VS Code. Its reactivity encourages dynamic, interactive elements, because marimo is both a notebook environment and a library of UI components. The [AnyWidget](https://anywidget.dev/) protocol makes it relatively straightforward to extend marimo with custom widgets.

marimo does not allow out-of-order cell execution, which eliminates a common class of "it worked on my machine" complaints. And since a marimo notebook is a valid Python file, it integrates naturally with other Python tools and with version control. Instructors benefit from having prose and code examples in the same file, which helps keep explanations in sync with the code they describe. marimo also offers configurable integration with AI tools.

### Why *Not* marimo?

marimo is not yet as widely known as Jupyter, so your institution's IT department may not already support it. Auto-grading integration is not yet available in a mature form (but we're working on it), and multi-chapter books are not yet supported. Some quirks in how marimo handles scope may make it a less natural fit for an introductory CS course; see "Things to Watch Out For" below.

## molab

molab is marimo's free cloud-hosted notebook service, available at [molab.marimo.io](https://molab.marimo.io/notebooks). It is the easiest way for students to get started because it requires no local installation: marimo is accessible as a self-contained web application, comparable in experience to Google Colab. Notebooks created on molab are public but not discoverable by default, and can be shared with others by URL. Students can download their notebooks as `.py`, `.ipynb`, or PDF files, which makes submission to grading systems such as Gradescope straightforward.

molab can also preview notebooks hosted on GitHub. The service provides a stable URL for a notebook that stays current as the notebook changes, so students always see the latest version. From the preview page, students can fork the notebook into their own workspace.

Existing Jupyter notebooks can be converted to marimo notebooks automatically with `uvx marimo convert my_notebook.ipynb -o my_notebook.py`.

## Ways to Teach With marimo

At a high level, there are (at least) four ways to teach with marimo notebooks:

1.  Learners can follow along with the lesson when the code already present in the notebook.

1.  Notebooks can be used as assignments (i.e., "fill in these cells").

1.  Notebooks be used as apps so that learners can explore data rather than, or as well as, writing code.

1.  Learners can create notebooks from scratch as lab reports, which most closely models real-world use.

## Things to Watch Out For

Variable names
:   Underscore-prefixed variable names in marimo have a meaning different from their common usage in Python and require some understanding of lexical scope. The recommended remedy is a functions-early teaching methodology, which has a sound pedagogical basis and prepares learners for idiomatic Python.

Image files
:   For security reasons, marimo requires local image files to be placed in a folder named `public` directly below the directory from which the notebook is launched, and to be referenced in Markdown as `/public/image.ext`. This means it matters where the notebook is started. The `mo.image` function works around this restriction but cannot be embedded inside a Markdown string.

Testing
:   [Using pytest in marimo](https://docs.marimo.io/guides/testing/pytest/#testing-in-notebook) is straightforward, provided that each test cell contains only tests and nothing else.

Math rendering
:   marimo uses [KaTeX](https://katex.org/) rather than [MathJax](https://www.mathjax.org/) for rendering mathematics. The two systems are largely compatible but differ in some commands and environments. See the appendix for details.

## Pedagogical Patterns

Much of this section is inspired by or taken from
[*Teaching and Learning with Jupyter*](https://jupyter4edu.github.io/jupyter-edu-book/).
We are grateful to its authors for making their work available under an open license.

### Shift-Enter

Learners start with a complete notebook and re-execute the cells in order, optionally filling in prose cells with analysis or description. This pattern is well suited to introducing new topics or checking understanding through warmup exercises, and works with any audience in a synchronous setting. It gives learners a working example immediately, though engagement tends to be low because learners are not yet making decisions.

### Fill in the Blanks

Some code cells are provided complete; learners must complete the rest. This reduces cognitive load by directing attention to a single concept, such as filtering a dataset, and works for any audience in assignments and lab sessions. The risk is that learners delegate the task to an AI tool, and the difficulty of the blanks can be hard to calibrate for a mixed-ability group.

### Tweak and Twiddle

Learners start with a complete, working notebook and are asked to alter parameters in order to achieve a specified goal. This pattern supports compare-and-contrast exercises and the acquisition of domain knowledge. It is particularly effective for learners who have domain knowledge but little programming experience, in fixed-time workshop exercises or pair programming sessions. It helps learners overcome anxiety about code. The main difficulties are that learners may not know where to start, or may spend time following unproductive tangents.

### Notebook as App

The notebook is presented as an interactive dashboard, with prose kept in a separate document so the interface looks like a standalone application. This pattern is designed for non-programmers exploring datasets. It can replace slides in a lecture if the instructor knows the material well enough to navigate live, or it can be used after a physical lab experiment for data analysis. It requires less effort to build than a custom UI, but it demands thorough testing and does not develop learners' programming skills.

### Top-Down Delivery

Learners are given just enough control to reach a motivating result quickly. The goal is engagement on the first day of a course or workshop. This pattern works for any audience but is most effective with learners who have limited programming experience, in tutorials and synchronous workshops. Student engagement is the main advantage; the main challenge is finding the right level of detail for a group with mixed abilities.

### Coding as Translation

Learners convert prose to code, or code to prose. The purpose is to connect concepts to implementations and implementations to concepts. It is well suited to learners who understand theory but struggle with coding, or the reverse. A notebook with scaffolding text and possibly some pre-written code works well as the format. The barrier to entry is low for learners with limited programming background; the challenge, again, is calibrating difficulty for a mixed-ability group.

### Symbolic Math

Learners use SymPy to do symbolic mathematics inside the notebook, extending the coding-as-translation pattern to include converting mathematical expressions to code or code to mathematical expressions. This works well for STEM students interested in theory and fits any format. It introduces another real-world tool, but SymPy's syntax is yet another thing to learn on top of the mathematics itself.

### Numerical Methods and Simulation

Learners use calculation or simulation rather than closed-form analysis to make a concept tangible before the mathematical abstraction is introduced. This requires some programming skill and fits any format. Going from specific to general is often more engaging and approachable than the reverse, but debugging numerical code can be difficult.

### Learn an API

A key library or API is introduced example by example, in order of increasing complexity or decreasing frequency of use. The purpose is to direct learner attention toward tools they will use in other parts of the course. This works for learners with some programming skill and patience. It guides learning in a sensible order, which AI tools sometimes struggle to provide on their own. The risk is that learners lose sight of the larger goal, or prefer to ask an AI for help as needed rather than building systematic knowledge.

### Choose Your Data

Learners replace the dataset in a provided notebook with one of their own choosing, possibly making some modifications to the code. The goal is engagement through personal relevance, and it works well for learners with a specific domain interest such as sports analytics. A common structure is a shared first half followed by independent exploration, sometimes leading to presentations. It improves self-efficacy, but learners may struggle to find suitable data, encounter data that is too messy to work with, or have interests that do not overlap enough for a shared debrief.

### Test-Driven Learning

The instructor provides a notebook full of tests; learners must write code that makes those tests pass. This teaches learners to think in terms of a specification. It works for learners who want firm goalposts and fits homework exercises well. The task is well-defined and easy to stay focused on, but it is very easy for learners to have an AI generate the code without understanding it.

A useful pattern is to place a stub function in one cell and pytest tests in a separate cell. Because of marimo's reactive execution, every time the learner edits their implementation the tests rerun automatically, giving immediate feedback on correctness without giving away the answer. For example, the stub cell might contain:

```python
def add(x, y):
    """Return the sum of x and y."""
    # your code here
    pass
```

and the test cell:

```python
def test_add_integers():
    assert add(5, 6) == 11

def test_add_floats():
    assert isinstance(add(4, 2.1), float)
```

### Bug Hunt

Learners are given a notebook with one or more bugs, which may include misleading prose. The purpose is to develop critical reading skills, which is especially important for learners who regularly use AI tools. It requires enough programming experience to debug systematically and works well as a homework exercise. Some learners find the detective aspect genuinely engaging, and the skill is extremely valuable. The main challenges are calibrating bug difficulty and helping learners know when they are done.

### Adversarial Programming

Given a notebook full of code, learners write tests designed to break it. This is the reverse of the bug hunt. The purpose is to develop critical thinking, and it requires the same level of programming experience. It works well as a homework exercise. It helps learners appreciate the difficulty of writing robust code and sharpens their debugging skills, but learners sometimes find repetitive ways to break the code rather than probing for distinct failure modes.

## Appendix: Learner Personas

The three profiles below outlines who we're trying to help and how.

### Anya Academic

Anya is a biology professor at a mid-sized state university who teaches undergraduate microbiology and biostatistics, both of which emphasize data management and visualization. She used R for fifteen years before switching to Python three years ago, largely through self-teaching, and she routinely remixes teaching material she finds online.

She wants to equip her students with modern skills, including AI-related tools, both because she sees them as important and because she hopes they will increase student engagement. She is also looking for more recognition at her university for her teaching work, which she believes is more likely to come from publishable innovation than from high student evaluations. She would like to restore the level of student engagement she saw before the pandemic, which she attributes in part to the shift toward online education reducing social cohesion in her cohorts.

Her main concern is tool setup and maintenance overhead. She does not have time to rewrite courses wholesale, so she will only migrate to a new tool if there is an incremental path that allows her to back out if things are not working. Her department has only two IT staff, and nothing at her university can move beyond a pilot without integrating with the institution's learning management system.

### Ellis Engineer

Ellis is a senior undergraduate in mechanical engineering who has just returned from their third co-op placement and is very interested in drones. They used Jupyter notebooks with Google Colab in their second semester and are comfortable with NumPy and Altair, though they have done roughly as many courses in MATLAB and AutoCAD as in Python.

Ellis wants to build an impressive senior project to strengthen their graduate school application. They have seen custom widgets in notebooks and are willing to invest time in learning to build one with AI support. They also want to explore small-craft aerodynamics, particularly feedback stability problems, both as a personal interest and as a way to become known in the drone community.

Having spent a period convinced that Lisp was the language of the future, Ellis is now cautious about investing heavily in new technologies purely because they seem exciting.

### Nang Newbie

Nang is an undergraduate business student who decided against minoring in computer science because he believes AI will take most programming jobs. He chooses courses, tools, and interests largely based on what he reads about future employer demand, and he routinely uses ChatGPT for help with homework. He used Scratch in middle school and covered HTML and basic Python in a high school CS course. He just finished an introductory statistics course that used Pandas, which he enjoyed enough to sign up for the follow-on course.

He wants to complete assignments more quickly and with less effort. He also wants to learn how to explore and analyze sports statistics for fun, as a keen basketball fan, and to share findings with other fans through online forums.

Nang is taking five courses and volunteering with two campus clubs, one for his CV and one out of genuine passion for basketball, so he is chronically over-committed.

## Appendix: KaTeX vs. MathJax

marimo uses [KaTeX](https://katex.org/) for rendering math rather than [MathJax](https://www.mathjax.org/). KaTeX is faster and has slightly narrower coverage, and it fails silently when it encounters an unsupported command.

### Use Raw Strings

LaTeX lives in Python strings in marimo, so use `r"..."` to preserve backslashes:

```python
mo.md(r"$\frac{1}{2}$")   # correct
mo.md("$\frac{1}{2}$")    # wrong: \f is a form-feed character
```

### MathJax to KaTeX

| Category | MathJax | KaTeX |
| --- | --- | --- |
| Text | `\mbox`, `\bbox` | `\text{}` |
| Text style | `\textsc`, `\textsl` | `\text{}` |
| Environments | `\begin{eqnarray}` | `\begin{align}` |
|  | `\begin{multline}` | `\begin{gather}` |
| References | `\label`, `\eqref`, `\ref` | `\tag{}` for manual numbering |
| Arrays | `\cline`, `\multicolumn`, `\hfill`, `\vline` | not supported |
| Macros | `\DeclareMathOperator` | `\operatorname{}` inline |
|  | `\newenvironment` | not supported |
| Spacing | `\mspace`, `\setlength`, `\strut`, `\rotatebox` | not supported |
| Conditionals | `\if`, `\else`, `\fi`, `\ifx` | not supported |

The following commands do work in KaTeX despite claims to the contrary in some older references: `\newcommand`, `\def`, `\hbox`, `\hskip`, `\cal`, `\pmb`, `\begin{equation}`, `\begin{split}`, `\operatorname*`.

### Shared Macros Across Cells

`\newcommand` works inline within a single cell. For macros that need to be available across multiple cells, use `mo.latex(filename="macros.tex")` in the same cell as your `import marimo` statement.

### Migration Checklist

1. Find and replace `\mbox{` with `\text{`.
2. Wrap all LaTeX strings in raw string literals (`r"..."`).
3. Replace `\begin{eqnarray}` with `\begin{align}`.
4. Replace `\DeclareMathOperator` with `\operatorname{}`.
5. Remove `\label` and `\eqref`; use `\tag{}` where manual numbering is needed.
6. Verify the output visually, since KaTeX fails silently.

### References

- [KaTeX Support Table](https://katex.org/docs/support_table) — definitive command lookup
- [KaTeX Unsupported Features](https://github.com/KaTeX/KaTeX/wiki/Things-that-KaTeX-does-not-(yet)-support)
