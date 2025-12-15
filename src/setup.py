from setuptools import setup, find_packages
import os

with open('version.txt', 'r') as version_file:
    version = version_file.read().strip()

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join(path, filename))
    return paths

extra_files = package_files('sonarkan')

setup(
    name="SonarKAN",
    version=version,
    url="https://soundai.com",
    author="Chen Xiaoliang",
    author_email="chenxiaoliang@soundai.com",
    description="Sonar Kolmogorov–Arnold Network Based on GPU or CPU.",
    license="Apache License",    
    packages=find_packages(),  # Automatically find and include all packages
    include_package_data=True,  # Include non-code files specified in MANIFEST.in
    package_data={'': extra_files},  # Including extra files
    keywords=[
        "SonarKAN",      
        "Sonar Kolmogorov–Arnold Network",        
        "sonar",
        "kan",
    ], 
    install_requires=[
        'numpy',
        'PyYAML',
        'torch',
    ],
    classifiers=[
        # Add classifiers to categorize your project
        # See https://pypi.org/classifiers/ for a full list
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.10",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "License :: OSI Approved :: Software License",
        "Topic :: Software Development :: Libraries :: Python Modules",        
    ],
)
