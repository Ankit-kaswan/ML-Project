from setuptools import find_packages, setup


def get_requirements(file_path: str, exclude: list = None):
    """Reads dependencies from the given file and returns a list, excluding specified items."""
    if exclude is None:
        exclude = ["-e ."]  # Default exclusions

    with open(file_path) as file:
        requirements = [line.strip() for line in file if line.strip() and line.strip() not in exclude]

    return requirements


setup(
    name="mlproject",
    version="0.0.1",
    author="Ankit",
    author_email="ankit.iitd2014@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt", exclude=["-e ."]),
)
