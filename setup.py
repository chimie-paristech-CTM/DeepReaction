from setuptools import setup, find_packages

setup(
    name="my_package",                  # 软件包名称
    version="0.1.0",
    description="A package for reaction data processing and model training",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Name",
    author_email=".email@example.com",
    url="https://github.com/username/my_package",  # 项目主页
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "torch_geometric",
        "scikit-learn",
        "pytorch_lightning",
        # 添加其它依赖项
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)
