from setuptools import setup, find_packages

setup(
    name="RS",
    version="0.1",
    packages=["src"],
    entry_points={
        'console_scripts': [
            'rs-cli = src.__main__:app'
        ],
    },
)
