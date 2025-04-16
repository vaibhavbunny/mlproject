from setuptools import find_packages, setup

from typing import List

HYPEN_E_DOT = '-e . '
def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]
        
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements

install_requires = get_requirements('requirements.txt')
print("Requirements being installed:", install_requires)
print("Type of install_requires:", type(install_requires))

setup(
    name='mlproject',
    version='0.0.1',
    author='Vaibhav',
    author_email='kalevaibhav2903@gmail.com',
    packages=find_packages(),
    install_requires=install_requires
)
