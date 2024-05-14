from setuptools import find_packages,setup
from typing import List
def get_requirements(req_path):
    with open(req_path,'r') as f:
        requirements = f.readlines()
        requirements = [req.replace('\n','') for req in requirements]

        requirements.remove('-e .')

        return requirements



setup(
    name='Face-Based-Attendance-System',
    version='0.0.1',
    author='J j',
    author_email = 'georgejoshi10@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
)