from setuptools import find_packages,setup
from typing import List
def get_requirements(file_path:str)->List[str]:
    '''
    this will return list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements = [r.strip() for r in requirements if r.strip() != "" and not r.startswith("-e")]
        return requirements

setup(
    name='mlproject',
    version='0.0.1',
    author="Sachin Venkat",
    author_email='tsachinvenkat52@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)