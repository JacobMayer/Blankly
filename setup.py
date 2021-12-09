from setuptools import setup, find_packages

setup(
    name='trading_bot',
    version='1.2.0',
    packages=find_packages(),

    install_requires=[
        'gym>=0.12.5',
        'numpy>=1.16.4',
        'pandas>=0.24.2',
        'matplotlib>=3.0.3'
    ],

    package_data={
        'trading_bot': ['datasets/data/*']
    }
)