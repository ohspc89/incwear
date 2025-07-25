import setuptools

with open("README.md") as file:
    read_me_description = file.read()

setuptools.setup(name="incwear",
        version="0.1",
        author="Jinseok Oh, PhD",
        author_email="jinseok.oh@pm.me",
        description="A package to analyze wearable sensor data",
        long_description=read_me_description,
        long_description_content_type="text/markdown",
        url= "github.com/ohspc89/incwear.git",
        packages=['incwear'],
        classifiers=["Programming Language :: Python :: 3", "License :: OSI Approved :: MIT License", "Operating System :: OS Independent",],
        python_requires='>=3.10',)
