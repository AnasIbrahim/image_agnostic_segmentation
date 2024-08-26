from setuptools import setup, find_packages

setup(
    name='dounseen',  # Replace with your actual package name
    version='0.1.0',
    description='A brief description of your package',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Your Name',  # Replace with your name
    author_email='your.email@example.com',  # Replace with your email
    packages=find_packages(where='.'),  # Automatically find packages in the current directory
    install_requires=[
        'torch==2.3.1',  # Specify your dependencies here
    ],
    python_requires='>=3.6',  # Specify the Python version requirement
)

