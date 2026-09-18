k# AI Engineering Platform

A modular Python-based learning and development platform designed to demonstrate practical Software Engineering principles, Linux command-line usage, Git/GitHub workflows, Clean Code, Test-Driven Development (TDD), and structured project development.

## Project Overview

The AI Engineering Platform is a command-line based software project developed from scratch using Python.

The project was built incrementally using a structured development workflow that demonstrates how software engineering concepts can be applied to a realistic project.

The current version focuses on project management functionality with persistent JSON storage.

## Main Features

- Project creation
- Project listing
- Project validation
- JSON-based persistence
- Command-line interface (CLI)
- Service layer
- Repository layer
- Domain model
- Output formatting
- Automated testing
- Input validation
- Clean and modular architecture

## Technologies

- Python 3.13+
- Pytest
- Git
- GitHub
- Linux / Bash
- JSON
- setuptools
- Virtual Environment

## Project Architecture

```text
ai-engineering-platform/
│
├── data/
│   └── projects.json
│
├── src/
│   └── ai_platform/
│       ├── cli/
│       │   └── main.py
│       │
│       ├── core/
│       │   └── application.py
│       │
│       └── projects/
│           ├── project.py
│           ├── registry.py
│           ├── repository.py
│           ├── service.py
│           └── formatter.py
│
├── tests/
│   ├── test_application.py
│   ├── test_cli.py
│   ├── test_project.py
│   ├── test_project_formatter.py
│   ├── test_project_registry.py
│   ├── test_project_repository.py
│   └── test_project_service.py
│
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Software Design

The project separates responsibilities into several layers.

### Domain Layer

Contains the `Project` model and its validation rules.

### Service Layer

Contains project management operations such as creating and listing projects.

### Repository Layer

Handles persistent storage using JSON files.

### CLI Layer

Provides the command-line interface through which users interact with the system.

### Formatter

Responsible for converting project data into readable CLI output.

## CLI Usage

### List Projects

```bash
python -m ai_platform.cli.main project list
```

Example output:

```text
Projects:
- Persistent Project: Saved in JSON
```

### Add a Project

```bash
python -m ai_platform.cli.main project add --name "My Project" --description "My project description"
```

Example output:

```text
Project added: My Project
```

## Testing

The project uses Pytest for automated testing.

Run all tests:

```bash
python -m pytest
```

The current test suite contains 25 passing tests covering:

- Application initialization
- Project creation
- Project validation
- Whitespace validation
- Project registry
- Project service
- JSON repository
- CLI commands
- CLI integration
- Project formatting
- Persistence

Expected result:

```text
25 passed
```

## Test-Driven Development

Development followed the TDD cycle:

```text
RED
 ↓
Write a failing test
 ↓
GREEN
 ↓
Implement the minimum solution
 ↓
REFACTOR
 ↓
Improve the implementation
```

Examples implemented using this approach include:

- Project name validation
- Project description validation
- Whitespace validation
- Project creation
- Repository persistence
- CLI integration

## Clean Code Practices

The project applies several Clean Code principles:

- Meaningful names
- Small focused functions
- Single responsibility
- Separation of concerns
- Type hints
- Docstrings
- Consistent formatting
- Validation at the domain level
- Modular architecture
- Automated tests
- Meaningful Git commits

## Linux and Bash

The project was developed in a Linux environment using the command line.

The development process included practical usage of:

```text
pwd
ls
cd
mkdir
touch
cat
nano
python
pip
git
ssh
```

A Python virtual environment was also created and used:

```text
.venv/
```

## Git Workflow

The project was developed using Git with meaningful commits and feature branches.

Main branch:

```text
main
```

Feature branch:

```text
feature/project-core
```

Examples of commits:

```text
chore: initialize project structure
feat: add application core
test: add application initialization test
chore: configure Python package
feat: add project domain model
feat: add project registry
feat: add command line interface
feat: add project output formatter
feat: add project JSON repository
refactor: connect project service to repository
feat: persist projects through cli
feat: validate project name
feat: validate project description
test: cover whitespace-only project name
test: cover whitespace-only project description
```

## GitHub

The project is hosted publicly on GitHub.

Repository:

https://github.com/F13342/ai-engineering-platform

## Development Methodology

The project was developed incrementally rather than being created as one large implementation.

The workflow was:

```text
Project Initialization
        ↓
Python Package Setup
        ↓
Application Core
        ↓
Domain Model
        ↓
Tests
        ↓
Service Layer
        ↓
CLI
        ↓
Repository
        ↓
JSON Persistence
        ↓
Validation
        ↓
Refactoring
        ↓
Integration Testing
        ↓
Git / GitHub
```

## Development Environment

Operating System:

```text
Kali Linux
```

Python:

```text
Python 3.13+
```

Environment:

```text
Python Virtual Environment (.venv)
```

Version Control:

```text
Git + GitHub
```

## Current Status

The current version provides a functional project-management foundation that can be extended into a larger AI Engineering platform.

Future modules may include:

- Dataset management
- Experiment tracking
- Model management
- AI/ML project templates
- Configuration management
- Model evaluation
- AI engineering workflows

## Learning Objectives Demonstrated

This project demonstrates practical application of:

1. Python programming
2. Linux command-line usage
3. Bash commands
4. Git version control
5. Git branching
6. Git commits
7. GitHub repository management
8. SSH authentication
9. Clean Code principles
10. Test-Driven Development
11. Software architecture
12. Separation of concerns
13. Automated testing
14. JSON persistence
15. CLI application development

## Author

**Fuad Alnhari**

Computer Science Student

GitHub:

https://github.com/F13342
