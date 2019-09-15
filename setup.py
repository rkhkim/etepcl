from setuptools import setup

setup(
    name = 'e2ep',
    packages = ['e2ep'],
    entry_points = {
            "console_scripts": ['e2ep = e2ep.cli:main']
        },
    version = '1.0',
    description = 'Command line end to end processing',
    author = 'Anmol Warman, Ryan Kim, Shawn Chao, Danny Guo',
    url = 'https://github.com/anmolwarman/e2ep',
    keywords = ['machine learning', 'ai', 'neural network'],
    setup_requires=["numpy"],
    install_requires=[
        'numpy',
        'scipy',
        'scikit-learn',
        'pyparsing',
        'matplotlib',
        'click'
    ],
)
