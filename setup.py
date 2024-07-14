from setuptools import setup, find_packages

setup(
    name='imdb_analysis',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'argparse'
    ],
    entry_points={
        'console_scripts': [
            'launch_analysis = launch_notebook:main'
        ]
    },
)
