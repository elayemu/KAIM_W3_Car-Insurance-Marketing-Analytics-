# Insurance Planning and Data Analysis

Welcome to the Insurance Planning and Data Analysis project! This initiative is dedicated to improving marketing strategies and identifying low-risk targets for premium adjustments in the insurance sector. We employ Data Version Control (DVC) to manage and track changes in our datasets, enabling effective collaboration.

## Data Version Control (DVC)

Data Version Control (DVC) is a free, open-source tool designed for managing large datasets, automating machine learning (ML) processes, and enhancing experiment management. By integrating DVC with Git, we ensure reproducibility, efficient data management, and secure collaboration.

### Key Features of DVC:
- **Codification**: Documents all aspects of the ML project (data versions, ML pipelines, and experiments) in human-readable metafiles, following industry best practices and established engineering tools.
- **Versioning**: Utilizes Git for versioning and sharing the complete ML project, which includes source code, configurations, and data assets. DVC metafiles act as stand-ins for the actual data files.
- **Secure Collaboration**: Allows for controlled project access, facilitating secure sharing with selected collaborators.

## Project Structure

The project structure is organized as follows:

```
├── .vscode/
│   └── settings.json
├── .github/
│   └── workflows/
│       └── unittests.yml
├── .gitignore
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   └── utils.py
├── notebooks/
│   ├── __init__.py
│   └── insurance_EDA.ipynb
├── tests/
│   ├── __init__.py
└── scripts/
    ├── __init__.py
    └── README.md
