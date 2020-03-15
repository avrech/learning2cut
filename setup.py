import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="learning2cut",  # Replace with your own username
    version="0.0.1",
    author="Avrech Ben-david",
    author_email="avrech@campus.technion.ac.il",
    description="Reinforcement learning for cut selection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/avrech/learning2cut",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Linux",
    ],
    python_requires='>=3.6',
)