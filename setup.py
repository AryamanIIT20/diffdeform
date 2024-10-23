from setuptools import setup, find_packages

setup(
    name='diffdeform',             # Package name
    version='0.1.0',               # Version number
    description='A package for image registration and deformation',
    author='Aryaman Vikram Todi',
    author_email='todi.aryaman@gmail.com',
    url='https://github.com/AryamanIIT20/diffdeform',  # Optional project URL
    packages=find_packages(),      # Automatically find packages in your directory
    install_requires=[
        'torch>=1.9.0',            # List any dependencies your package needs
        'numpy>=1.21.0',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Change as needed
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Minimum Python version required
)
