# setup.py

from setuptools import setup, find_packages

setup(
    name='lero',          # 包名称，建议与包目录同名
    packages=find_packages(),      # 自动发现所有包和子包
    python_requires='>=3.6',       # Python 版本要求
    install_requires=[],           # 依赖的第三方库列表，可从 requirements.txt 中读取
)
