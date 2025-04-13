from setuptools import setup, find_packages

setup(
    name="car_price_checker_api",
    version="0.1.0",
    description="Pipeline project for car price checking",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Rubén López",
    author_email="lopezrbn@gmail.com",
    license="MIT",
    packages=find_packages(),         # Automatically detects packages (folders with __init__.py)
    python_requires=">=3.7",
    install_requires=[
        "category_encoders",
        "pandas",
        "numpy",
        "scikit-learn",
        "sqlalchemy",
        "psycopg2-binary",
        "xgboost",
        "lightgbm",
        "pyarrow",
        "joblib",
        "fastapi",
        "gunicorn",
        "uvicorn",
        "requests",
        "beautifulsoup4",
    ],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
