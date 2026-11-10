from setuptools import setup, find_packages

setup(
    name='iges_sentence_splitter',
    version='0.2.1',
    description='A package for sentence splitting using a pre-trained transformer model.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Kathryn Chapman',
    author_email='kathryn.chapman@iges.com',
    # url='https://github.com/yourusername/sentence_splitter',
    packages=find_packages(),
    install_requires=[
        'torch',
        'transformers',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
