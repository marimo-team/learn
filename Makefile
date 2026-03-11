ROOT := .
SITE := _site
TMP := ./tmp
LESSON_DATA := ${TMP}/lessons.json
TEMPLATES := $(wildcard templates/*.html)

NOTEBOOK_INDEX := $(wildcard */index.md)
NOTEBOOK_DIR := $(patsubst %/index.md,%,${NOTEBOOK_INDEX})
NOTEBOOK_SRC := $(foreach dir,$(NOTEBOOK_DIR),$(wildcard $(dir)/??_*.py))
NOTEBOOK_OUT := $(patsubst %.py,${SITE}/%.html,$(NOTEBOOK_SRC))

DATABASES := \
sql/public/lab.db \
sql/public/penguins.db \
sql/public/survey.db

MARIMO := uv run marimo
PYTHON := uv run python

# Default target
all : commands

## commands : show all commands
commands:
	@grep -h -E '^##' ${MAKEFILE_LIST} | sed -e 's/## //g' | column -t -s ':'

## install: install required packages
install:
	uv pip install -r requirements.txt

## check: run all simple checks
check:
	-@make check_empty
	-@make check_titles
	-@make check_typos
	-@make check_packages

## check_exec: run notebooks to check for runtime errors
check_exec:
	@if [ -z "$(NOTEBOOKS)" ]; then \
		bash bin/run_notebooks.sh $(NOTEBOOK_SRC); \
	else \
		bash bin/run_notebooks.sh $(NOTEBOOKS); \
	fi

## build: build website
build: ${LESSON_DATA} ${NOTEBOOK_OUT} ${TEMPLATES}
	${PYTHON} bin/build.py --root ${ROOT} --output ${SITE} --data ${LESSON_DATA}

## links: check links locally (while 'make serve')
links:
	linkchecker -F text http://localhost:8000

## serve: run local web server without rebuilding
serve:
	${PYTHON} -m http.server --directory ${SITE}

## databases: rebuild datasets for SQL lessons
databases: ${DATABASES}

## ---: ---

## clean: clean up stray files
clean:
	@find . -name '*~' -exec rm {} +
	@find . -name '.DS_Store' -exec rm {} +
	@rm -rf ${TMP}
	@rm -f log_data_filtered*.*

## check_empty: check for empty cells
check_empty:
	@${PYTHON} bin/check_empty_cells.py

## check_titles: check for missing titles in notebooks
check_titles:
	@${PYTHON} bin/check_missing_titles.py

## check_packages: check for inconsistent package versions across notebooks
check_packages:
	@if [ -z "$(NOTEBOOKS)" ]; then \
		${PYTHON} bin/check_notebook_packages.py $(NOTEBOOK_SRC); \
	else \
		${PYTHON} bin/check_notebook_packages.py $(NOTEBOOKS); \
	fi

## check_typos: check for typos
check_typos:
	@typos ${TEMPLATES} ${NOTEBOOK_INDEX} ${NOTEBOOK_SRC}

## extract: extract lesson data
extract: ${LESSON_DATA}

#
# subsidiary targets
#

tmp/lessons.json: $(NOTEBOOK_INDEX)
	${PYTHON} bin/extract.py --root ${ROOT} --data ${LESSON_DATA}

${SITE}/%.html: %.py
	${MARIMO} export html-wasm --force --mode edit $< -o $@ --sandbox

sql/public/lab.db: bin/create_sql_lab.sql
	@rm -f $@
	@mkdir -p sql/public
	sqlite3 $@ < $<

sql/public/penguins.db: bin/create_sql_penguins.py data/penguins.csv
	@rm -f $@
	@mkdir -p sql/public
	${PYTHON} $< data/penguins.csv $@

sql/public/survey.db: bin/create_sql_survey.py
	@rm -f $@
	@mkdir -p sql/public
	${PYTHON} $< $@ 192837
