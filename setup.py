import setuptools

setuptools.setup(name="nn-additional-losses",
                 version="1.0",
                 url="https://github.com/abhi4ssj/nn-additional-losses",
                 author="Shayan Ahmad Siddiqui",
                 author_email="shayan.siddiqui89@gmail.com",
                 description="Contains additional losses which are still not part of pytorch standard library",
                 packages=setuptools.find_packages(),
                 install_requires=[torch, numpy])