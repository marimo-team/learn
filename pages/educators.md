---
title: marimo for Educators
---

## Introduction

- what *is* a notebook?
    - *literate programming* mixes prose and software in a single "runnable paper"
    - each *cell* is prose or software
    - prose typically written in Markdown
    - software written in whatever programming languages the notebook supports
    - software's output displayed in the notebook as well
- why notebooks for everyday work?
    - easier to understand (think about the way textbooks present material)
    - improves reproducibility
    - [GVW: if we emphasize embedded AI] keep track of what you asked for as well as what you did
- why notebooks for learning?
    - more engaging than static material: learners are active users of material, not passive consumers, can experiment with settings, alter code, etc.
    - no installation required: notebooks can be hosted so learners don't have to struggle with the hard bits first (i.e., focus on learning rather than on the tool)
    - reproducibility helps collaboration as well [GVW: but we don’t support concurrent editing a la Google Docs, which some people will regard as table stakes]
    - less intimidating than jumping straight into scripting
    - introduces a real-world tool
    - [if we emphasize embedded AI] a natural way to bring LLMs into the classroom
- why notebooks for teaching?
    - all of the above…
    - create interactive lecture material in a single place
- why the marimo notebook?
    - open source
    - more than Notebook but not as intimidating as VS Code
    - reactivity allows for (encourages) dynamic, interactive elements
        - marimo is both a notebook and a library of UI elements
        - and [AnyWidget](https://anywidget.dev/) makes it relatively easy to extend [GVW: point at [faw](https://github.com/gvwilson/faw)]
    - doesn't allow out-of-order execution of cells, which reduces “worked for me” complaints
    - plays nicely with other Python tools (because a notebook is a Python file)
    - plays nicely with version control (same reason)
    - helps instructors keep their prose and examples in sync
    - configurable interaction with AI tools
    - [if we emphasize embedded AI] natural way to teaching prompting and review
- why *not* marimo?
    - not yet as widely known as Jupyter (i.e., your IT department may not already support it)
    - not yet integrated with auto-grading tools ([faw](https://github.com/gvwilson/faw) is a start, but we're waiting to see what you want)
    - doesn't yet support multi-notebook books
    - some quirks that might not make it the right tool for a CS-101 course (see below)

## Ways to Teach With marimo

- high level
    - follow along with lesson (code already present)
    - workbooks for assignments ("fill in these cells")
    - notebooks as apps (play with data rather than write code)
    - notebooks as lab reports (models real-world use)
- micro
    - scroll through a pre-executed notebook
    - step through a notebook by executing the cells in order
    - fill out details or values into a mostly complete notebook
    - tweak or fill in a notebook with some content
    - add content to a completely blank notebook
    - ask learners what to add *or* what's going to happen
    - ask AI to do something and then explore/correct/improve its output

## Things to Watch Out For

- Variable names
    - Underscored variable names are different from common usage, and require some understanding of scope
    - Solution is functions-early teaching methodology, which has a sound pedagogical basis
- Image files
    - For security reasons, marimo requires local image files to be in a folder called `public` below the directory the notebook is run from, and to be accessed in Markdown as `[alt text](/public/image.ext)`
    - Which means it’s important to launch the notebook from the right place
    - Can get around this using `mo.image` but that can’t be embedded in Markdown
- [Using pytest in marimo](https://docs.marimo.io/guides/testing/pytest/#testing-in-notebook) is straightforward as long as the cell *only* contains tests
- marimo uses [KaTeX](https://katex.org/) rather than [MathJax](https://www.mathjax.org/) for rendering math - see the appendix to this document for notes

## Pedagogical Patterns

### Shift-Enter

**Description:** Learner starts with complete notebook, re-executes cells; (possibly) fills in prose cells with analysis/description

**Use For:** Introduce new topics; check understanding (e.g., warmup exercise)

**Works For:** Any audience

**Format:** Synchronous

**Pro:** Gives learners a complete working example

**Con:** Low engagement

### Fill in the blanks

**Description:** Some code cells filled in, learner must complete

**Use For:** Reducing cognitive load

**Works For:** Any audience

**Format:** Assignments and labs

**Pro:** Focus attention on a specific concept (e.g., filtering data)

**Con:** “Just get AI to do it”; required work can be too easy or too hard

### Tweak and twiddle

**Description:** Learner starts with complete working notebook, is asked to alter parameters to achieve some goal

**Use For:** Compare and contrast; acquiring domain knowledge

**Works For:** Learners without programming experience (but requires some domain knowledge)

**Format:** Fixed-time workshop exercise; pair programming

**Pro:** Helps learners overcome code anxiety

**Con:** “Where do I start?” and going down rabbit holes

### Notebook as app

**Description:** Use notebook as interactive dashboard (note: usually keep prose in a separate document to make the dashboard look like an app)

**Use For:** Exploring datasets

**Works For:** Non-programmers

**Format:** Use instead of slides (but must know where you’re going); have learners suggest alternatives to explore; data analysis after (physical) lab experiment

**Pro:** Less effort to build than custom UI

**Con:** Requires testing; does not develop programming skills

### Top-down delivery

**Description:** Give learners just enough control to get to a motivating result quickly (“day one”)

**Use For:** Follow-along lectures

**Works For:** Any audience (but most engaging for people with low programming skills)

**Format:** Tutorials and workshops (synchronous)

**Pro:** Student engagement

**Con:** Hard to get the right level of detail for a mixed-ability audience

### Coding as translation

**Description:** Convert prose to code (or vice versa)

**Use For:** Connect concepts to implementation (and implementation to concepts)

**Works For:** Learners who understand theory but struggle with coding (or vice versa)

**Format:** Notebook with scaffolding text and possibly some (scaffolded) code

**Pro:** Low barrier to entry for learners with limited programming knowledge

**Con:** Hard to get the level right for mixed-ability audience

### Symbolic math

**Description:** Use SymPy for symbolic math in notebook

**Use For:** Extension of previous exercise: convert math to code or code to math

**Works For:** STEM students interested in theory

**Format:** Any

**Pro:** Introduce another real-world tool

**Con:** Math in SymPy is yet another thing to learn

### Numerical methods / simulation

**Description:** Use calculation or simulation instead of formulaic analysis

**Use For:** Make concepts tangible before introducing mathematical abstraction

**Works For:** Learners with some programming skill

**Format:** Any

**Pro:** Going from specific to general is often more engaging and approachable

**Con:** Requires programming skill; can be hard to debug

### Learn an API

**Description:** Introduce a key API example by example

**Use For:** Put focus on tools to be used in other places / lessons

**Works For:** Learners with some programming skill (and patience)

**Format:** Examples in order of increasing complexity or decreasing frequency of use

**Pro:** Guide learning in a sensible order (which AI sometimes struggles with)

**Con:** “Can’t see the forest for the trees”; learners may prefer just asking AI as needed

### Choose your data

**Description:** Replace the dataset used in a notebook with another one (which may require some modifications to code)

**Use For:** Engagement

**Works For:** Learners with specific domain interest (e.g., sports analytics)

**Format:** Common first half, learners explore on their own for second half; learners create presentations to share with others

**Pro:** Improves self-efficacy; leverages engagement with personal interests

**Con:** Can’t find data, data is too messy, learners’ interest don’t overlap

### Test-driven learning

**Description:** Instructor provides notebook full of tests; learners must write code to make those tests pass (e.g., handle messy data)

**Use For:** Think in terms of a spec

**Works For:** Learners who want firm goalposts

**Format:** Notebook full of test cases with empty cells (and function stubs) for code; works well for homework exercises

**Pro:** Helps learners stay focused on well-defined task

**Con:** Very easy to have AI generate the code without understanding it

### Bug hunt

**Description:** Give learners a notebook with one or more bugs (which can include misleading prose)

**Use For:** Developing critical reading skills (especially important for learners using AI)

**Works For:** Learners with enough programming experience to be able to debug systematically

**Format:** Works well as homework exercise

**Pro:** Some learners enjoy playing detective; extremely useful skill to learn

**Con:** Hard to calibrate bug difficulty to learner level; hard for learners to know when they’re done

### Adversarial programming

**Description:** Given a notebook full of code, write tests that break it (reverse of bug hunt)

**Use For:** Learning critical thinking

**Works For:** Learners with enough programming experience to be able to debug systematically

**Format:** Works well as homework exercise

**Pro:** Helps learners appreciate how hard it is to write robust code; improves their debugging skills

**Con:** Learners can break code in repetitive ways (e.g., provide several inputs that trigger the same flaw)

## Acknowledgments

Much of this is inspired by or taken from
[*Teaching and Learning with Jupyter*](https://jupyter4edu.github.io/jupyter-edu-book/).

## Appendix: Learner Personas

### Anya Academic

**Background:** Biology professor at mid-sized state university; teaches undergrad microbiology and biostatistics classes, both of which emphasize data management and visualization.

**Relevant Experience:** Used R for 15 years, switched to Python three years ago, mostly self-taught. Frequently remixes teaching material she finds online, particularly examples.

**Goals**

1. Wants to equip her students with modern skills, especially AI-related, both because she thinks they’re important and to increase student engagement.

2. Wants more recognition at her university for her teaching work, which she believes is more likely to come from publishable innovation than from high student evaluations.

3. Would like to get student engagement back to pre-COVID levels; she feels that today’s cohorts don’t know each other as well and aren’t as excited about material because of the shift to online education.

**Complications**

1. Is concerned about tool setup and maintenance overheads. Doesn't have time to completely rewrite courses, so will only move over if there's an incremental migration path that allows her to back out if thing don't appear to be working.

2. Anya's department has two overworked IT staff, and nothing at her university is allowed to go beyond the pilot phase if it doesn't integrate with the LMS somehow.

### Ellis Engineer

**Background:** Senior undergraduate in mechanical engineering who just returned to school from their third and final co-op placement. They are very excited about drones.

**Relevant Experience:** Used Jupyter notebooks with Colab in their second semester. They are comfortable with NumPy and Altair and has bumped into Pandas, but has done as many classes with MATLAB and AutoCAD as with Python.

**Goals**

1. Ellis wants to create an impressive senior project to secure themself a place in a good graduate program (which they think is essential to doing interesting work with drones). They have seen custom widgets in notebooks, and are willing to invest some time to learn how to build one with AI support.

2. They also want to explore small-craft aerodynamics, particularly feedback stability problems, out of personal interest and as a way to become part of the “serious” drone community.

**Complications**

Having spent several months convinced that Lisp was the language of the future, Ellis is leery of investing too much in new technologies just because they’re cool.

### Nang Newbie

**Background:** Undergraduate business student; decided not to minor in CS because "AI is going to eat all those jobs". Nang chooses courses, tools, and interests based primarily on what the web tells him potential future employers are going to look for. He routinely uses ChatGPT for help with homework.

**Relevant Experience:** Used Scratch in middle school and did one CS class in high school that covered HTML and a bit of Python. He just finished an intro stats class that used Pandas, which to his surprise he enjoyed enough to sign up for the sequel.

**Goals**

1. Nang wants to be able to do homework assignments more quickly and with less effort (hence his interest in ChatGPT).

2. He wants to learn how to explore and analyze sports statistics for fun (he's a keen basketball fan), and to share what he finds with like-minded fans through online forums.

**Complications**

Nang is taking five courses and volunteering with two campus clubs (one for the sake of his CV, and one because of his passion for basketball), so he is chronically over-committed.

## Appendix: KaTeX vs. MathJax

marimo uses [KaTeX](https://katex.org/) for rendering math (faster, slightly narrower coverage, silent errors) rather than [MathJax](https://www.mathjax.org/).

### Use raw strings

LaTeX lives in Python strings in marimo, so use `r"..."` to preserve backslashes:

```python
mo.md(r"$\\frac{1}{2}$")   # ✅
mo.md("$\\frac{1}{2}$")    # ❌ — \\f is a form-feed character
```

### MathJax → KaTeX

| Category | MathJax | KaTeX |
| --- | --- | --- |
| Text | `\\mbox`, `\\bbox` | `\\text{}` |
| Text style | `\\textsc`, `\\textsl` | `\\text{}` |
| Environments | `\\begin{eqnarray}` | `\\begin{align}` |
|  | `\\begin{multline}` | `\\begin{gather}` |
| References | `\\label`, `\\eqref`, `\\ref` | `\\tag{}` for manual numbering |
| Arrays | `\\cline`, `\\multicolumn`, `\\hfill`, `\\vline` | — |
| Macros | `\\DeclareMathOperator` | `\\operatorname{}` inline |
|  | `\\newenvironment` | — |
| Spacing | `\\mspace`, `\\setlength`, `\\strut`, `\\rotatebox` | — |
| Conditionals | `\\if`, `\\else`, `\\fi`, `\\ifx` | — |

These *do* work in KaTeX (despite outdated claims): `\\newcommand`, `\\def`, `\\hbox`, `\\hskip`, `\\cal`, `\\pmb`, `\\begin{equation}`, `\\begin{split}`, `\\operatorname*`.

### Shared macros across cells

`\\newcommand` works inline. For cross-cell reuse, use `mo.latex(filename="macros.tex")` in the same cell as `import marimo`.

### Migration checklist

1. Find-replace `\\mbox{` → `\\text{`
2. Use raw strings (`r"..."`)
3. Replace `\\begin{eqnarray}` → `\\begin{align}`
4. Replace `\\DeclareMathOperator` → `\\operatorname{}`
5. Remove `\\label`/`\\eqref` → use `\\tag{}` if needed
6. Visually verify — KaTeX fails silently

### References

- [KaTeX Support Table](https://katex.org/docs/support_table) — definitive command lookup
- [KaTeX Unsupported Features](https://github.com/KaTeX/KaTeX/wiki/Things-that-KaTeX-does-not-(yet)-support)