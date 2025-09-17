from setuptools import setup, find_packages

setup(
    name="pymoon",
    version="0.1.0",
    description="A package for moon and disk mask calculations.",
    author="Pererossello",
    packages=find_packages(),
    install_requires=[
        "numpy"
    ],
    python_requires=">=3.7",
)