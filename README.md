# AI Engineering Platform

A modular Python-based learning and development platform designed to demonstrate practical Software Engineering, Linux, Git/GitHub, Clean Code, Test-Driven Development (TDD), Data Engineering, and structured AI development workflows.

The platform is designed as a cumulative project where each learning module builds on the previous modules and prepares the foundation for future Machine Learning, Deep Learning, Generative AI, AI Agents, and AI Engineering modules.

---

## Project Overview

The AI Engineering Platform is a Python-based project developed incrementally using a structured software engineering workflow.

The project started as a software engineering foundation and was then extended with a multi-source Data Engineering pipeline.

The long-term architecture is designed around the following progression:

```text
Problem
   ↓
Data
   ↓
Data Engineering
   ↓
Machine Learning
   ↓
Deep Learning
   ↓
AI System
   ↓
AI Product
   ↓
Deployment
   ↓
Users
   ↓
Monetization
```

Each course and assignment can add a new module to the same platform instead of creating unrelated projects.

---

# Module 1: Software Engineering Foundation

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

---

## Technologies

- Python 3.13+
- Pytest
- Git
- GitHub
- Linux / Bash
- JSON
- setuptools
- Python Virtual Environment
- pandas
- requests
- SQLite

---

## Project Architecture

```text
ai-engineering-platform/
│
├── data/
│   ├── projects.json
│   ├── raw/
│   ├── processed/
│   └── rejected/
│
├── database/
│   └── students.db
│
├── logs/
│   └── pipeline.log
│
├── mock_api/
│   └── server.py
│
├── src/
│   └── ai_platform/
│       │
│       ├── cli/
│       │   └── main.py
│       │
│       ├── core/
│       │   └── application.py
│       │
│       ├── projects/
│       │   ├── project.py
│       │   ├── registry.py
│       │   ├── repository.py
│       │   ├── service.py
│       │   └── formatter.py
│       │
│       └── data_engineering/
│           │
│           ├── sources/
│           │   ├── csv_source.py
│           │   ├── api_source.py
│           │   └── database_source.py
│           │
│           ├── transformation/
│           │   ├── cleaner.py
│           │   ├── transformer.py
│           │   └── integration.py
│           │
│           ├── validation/
│           │   └── quality.py
│           │
│           ├── output/
│           │   └── csv_writer.py
│           │
│           └── utils/
│               └── logger.py
│
├── tests/
│
├── .gitignore
├── main.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

# Software Design

The project separates responsibilities into several layers.

## Domain Layer

Contains the `Project` model and its validation rules.

## Service Layer

Contains project management operations such as creating and listing projects.

## Repository Layer

Handles persistent storage using JSON files.

## CLI Layer

Provides the command-line interface through which users interact with the system.

## Formatter

Responsible for converting project data into readable CLI output.

---

# CLI Usage

## List Projects

```bash
python -m ai_platform.cli.main project list
```

Example output:

```text
Projects:
- Persistent Project: Saved in JSON
```

## Add a Project

```bash
python -m ai_platform.cli.main project add --name "My Project" --description "My project description"
```

Example output:

```text
Project added: My Project
```

---

# Testing

The project uses Pytest for automated testing.

Run all tests:

```bash
python -m pytest -q
```

The current complete test suite contains:

```text
27 passed
```

The tests cover:

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
- CSV loading
- REST API loading
- SQLite extraction
- Database integration

Expected result:

```text
27 passed
```

---

# Test-Driven Development

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
- Data source testing
- Database integration testing

---

# Clean Code Practices

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

---

# Linux and Bash

The project was initially developed in a Linux environment using the command line.

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

---

# Git Workflow

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

---

# GitHub

The project is hosted publicly on GitHub.

Repository:

https://github.com/F13342/ai-engineering-platform

---

# Module 2: Data Engineering

The platform was extended with a multi-source Data Engineering pipeline.

The purpose of this module is to demonstrate how data can be collected from multiple sources, cleaned, validated, integrated, transformed, and saved as a reliable dataset for future AI modules.

---

## Data Engineering Scenario

The current assignment uses student data from an educational institution.

The student information is collected from three different sources:

1. CSV file
2. REST API
3. SQLite database

The student scenario is used as the required educational test case for this module.

The architecture is designed to be reusable for future datasets and AI applications.

---

# Data Sources

## CSV Source

The CSV source contains:

```text
student_id
student_name
age
major
city
```

Example:

```text
1001,Ahmed Ali,21,Computer Science,Dhamar
```

---

## REST API Source

The REST API provides:

```text
student_id
gpa
attendance
status
```

The project includes a local mock REST API for testing and development.

Endpoint:

```text
http://localhost:8000/students
```

---

## SQLite Database

The SQLite database contains course and enrollment information.

Tables include:

```text
courses
enrollments
```

The database is connected to student records using:

```text
student_id
```

---

# Data Engineering Pipeline

The complete pipeline is:

```text
CSV ───────────────┐
                   │
REST API ──────────┼──→ Extract
                   │
SQLite ────────────┘
                         ↓
                      Clean
                         ↓
                    Integrate
                         ↓
                    Transform
                         ↓
                    Validate
                         ↓
              ┌──────────┴──────────┐
              ↓                     ↓
            Valid                Rejected
              ↓                     ↓
     final_dataset.csv      rejected_records.csv
```

---

# Extraction

The pipeline extracts data from:

```text
CSV
REST API
SQLite
```

Each source has a separate module.

```text
sources/
├── csv_source.py
├── api_source.py
└── database_source.py
```

This separation keeps source-specific logic outside the main pipeline.

---

# Data Cleaning

The cleaning stage performs:

- Column name normalization
- Removal of unnecessary spaces
- Text normalization
- Case normalization
- City normalization
- Major normalization
- Preparation for type conversion

Examples include:

```text
"  Mohammed Saleh  "
        ↓
" Mohammed Saleh "
        ↓
" Mohammed Saleh"
```

and:

```text
"computer science"
" COMPUTER SCIENCE "
        ↓
"Computer Science"
```

---

# Data Validation

The pipeline validates important data quality rules.

## Student ID

- Must not be missing
- Must be unique

## Age

Valid range:

```text
16 - 80
```

## GPA

Valid range:

```text
0 - 4
```

## Attendance

Valid range:

```text
0 - 100
```

## Score

Valid range:

```text
0 - 100
```

Invalid records are separated from valid records.

---

# Rejected Records

Records that fail validation are saved in:

```text
data/rejected/rejected_records.csv
```

Each rejected record includes a reason.

Examples:

```text
invalid age
invalid gpa
invalid attendance
invalid score
missing student_id
duplicate student_id
```

This provides traceability for data quality problems.

---

# Data Transformation

The transformation stage converts fields into appropriate data types and creates derived features.

Derived fields include:

## Performance Level

Based on GPA:

```text
Low
Medium
High
```

## Attendance Status

Based on attendance:

```text
Good
Low
```

The pipeline also calculates database-derived information such as:

```text
course_count
average_score
score
```

---

# Data Integration

The three data sources are integrated using:

```text
student_id
```

The integration process combines:

```text
CSV Student Information
        +
REST API Academic Information
        +
SQLite Course Information
```

into a unified dataset.

---

# Pipeline Outputs

The pipeline generates the following files:

```text
data/
│
├── raw/
│
├── processed/
│   └── final_dataset.csv
│
└── rejected/
    └── rejected_records.csv
```

Logging is stored in:

```text
logs/
└── pipeline.log
```

---

# Running the Data Engineering Pipeline

## Step 1: Start the Mock REST API

Run:

```bash
python mock_api/server.py
```

The API will run at:

```text
http://localhost:8000/students
```

Keep this terminal running.

---

## Step 2: Run the Pipeline

Open another terminal and run:

```bash
python main.py
```

The pipeline performs:

```text
Extract
 ↓
Clean
 ↓
Integrate
 ↓
Transform
 ↓
Validate
 ↓
Load
```

---

# Data Engineering Testing

Run the complete test suite:

```bash
python -m pytest -q
```

Current result:

```text
27 passed in 1.12s
```

All current tests pass successfully.

---

# Logging

The pipeline records execution information in:

```text
logs/pipeline.log
```

The log contains pipeline stages such as:

```text
Pipeline started
Extracting CSV data
Extracting API data
Extracting SQLite data
Cleaning source data
Integrating data sources
Transforming integrated data
Validating final records
Pipeline completed
```

---

# Requirements

The project uses the following main Python packages:

```text
pytest
pandas
requests
```

SQLite is provided through Python's standard library.

---

# Installation

Create and activate the virtual environment:

```bash
python -m venv .venv
```

Activate the environment according to the operating system.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

For development and testing:

```bash
python -m pip install -e ".[dev]"
```

---

# Integration With Future AI Modules

The Data Engineering module is not an isolated project.

It provides the data foundation for future AI modules in the same platform.

The planned architecture is:

```text
Software Engineering
        ↓
Data Engineering
        ↓
Machine Learning
        ↓
Advanced Machine Learning
        ↓
Deep Learning
        ↓
Computer Vision
        ↓
NLP + Transformers
        ↓
LLMs + Generative AI
        ↓
RAG + AI Agents + MCP
        ↓
AI Engineering + MLOps
        ↓
Cloud + Docker
        ↓
Testing + Cybersecurity
        ↓
System Design
        ↓
Product Development
        ↓
Complete AI Product
```

The processed data produced by Data Engineering can later become an input to Machine Learning and other AI modules.

---

# Cumulative Project Strategy

The platform follows a cumulative development strategy.

Instead of creating an independent project for every course, each course adds a new capability to the same platform.

```text
Course 1
Software Engineering
        ↓
Course 2
Data Engineering
        ↓
Course 3
Machine Learning
        ↓
Course 4
Deep Learning
        ↓
Course 5
Computer Vision
        ↓
Course 6
NLP / Transformers
        ↓
Course 7
LLMs / Generative AI
        ↓
Course 8
Agents / RAG / MCP
        ↓
AI Engineering
        ↓
Complete AI Product
```

This approach preserves previous work and creates a continuously growing AI Engineering system.

---

# Development Methodology

The project was developed incrementally rather than being created as one large implementation.

The development workflow includes:

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
Data Engineering
        ↓
CSV Extraction
        ↓
REST API Extraction
        ↓
SQLite Extraction
        ↓
Cleaning
        ↓
Integration
        ↓
Transformation
        ↓
Validation
        ↓
Output
        ↓
Testing
        ↓
Git / GitHub
```

---

# Development Environment

Operating System:

```text
Kali Linux / Windows
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

Database:

```text
SQLite
```

---

# Current Status

The platform currently contains:

```text
Software Engineering Foundation
        +
Data Engineering Pipeline
```

The Data Engineering pipeline successfully:

- Extracts data from CSV
- Extracts data from REST API
- Extracts data from SQLite
- Cleans data
- Normalizes data
- Integrates multiple sources
- Transforms data
- Validates records
- Rejects invalid records
- Generates a final dataset
- Generates rejected records
- Generates pipeline logs
- Runs automated tests

Current automated test result:

```text
27 passed
```

---

# Future Modules

Planned modules include:

- Data Engineering improvements
- Statistics
- Machine Learning
- Advanced Machine Learning
- Deep Learning
- Computer Vision
- NLP
- Transformers
- LLMs
- Generative AI
- RAG
- AI Agents
- MCP
- AI Engineering
- MLOps
- Cloud
- Docker
- Testing
- Cybersecurity
- System Design
- Product Development
- Model Evaluation
- Experiment Tracking
- Model Management
- AI Application Deployment

---

# Learning Objectives Demonstrated

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
16. CSV data processing
17. REST API integration
18. SQLite database processing
19. Data cleaning
20. Data validation
21. Data transformation
22. Multi-source data integration
23. Data quality management
24. Pipeline logging
25. Rejected-record handling

---

# Author

**Fuad Alnhari**

Computer Science Student

GitHub:

https://github.com/F13342