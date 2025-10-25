from setuptools import setup, find_packages

# 从 requirements.txt 读取依赖
with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="mini_webarena",
    version="0.1.0",
    author="Zhiheng",
    author_email="zhihenglyu.cs@gmail.com",
    description="A minimal local version of WebArena for autonomous browser agents.",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.8",
)
