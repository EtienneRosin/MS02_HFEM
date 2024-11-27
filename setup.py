from setuptools import setup, find_packages

setup(
    name="hfem",
    version="0.1",
    description="A library to deals with the Periodic Poisson equation (from a ENSTA work).",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author="Etienne Rosin",
    author_email="etienne.rosin.rosin@ensta-paris.fr",
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'pygmsh',
        'meshio',
        'scipy',
        'tqdm',
        'cmasher',
        'scienceplots'
    ],
    # package_data={
    #     '': ['*.json', '*.mplstyle', '*.cpp', '*.h'],
    # },
    include_package_data=True,
)


