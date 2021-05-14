import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="columnjump", # Replace with your own username
    version="1.0.0",
    author="Chris Willott",
    author_email="chriswillott1@gmail.com",
    description="Fix column jumps for JWST NIRISS detector",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/chriswillott/jwst/columnjump",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
