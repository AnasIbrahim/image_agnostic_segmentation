from setuptools import setup, find_packages

setup(
    name='dounseen',
    version='0.1.0',
    description='Object identification without training or fine-tuning',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Anas Gouda',
    author_email='anas.gouda@tu-dortmund.de',
    packages=find_packages(where='.'),
    install_requires=[
        'torch>=2.3.1',
        'opencv-python'
    ],
    python_requires='>=3.8',
)

