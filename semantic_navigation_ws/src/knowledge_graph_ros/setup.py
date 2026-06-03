from glob import glob

from setuptools import find_packages, setup

package_name = "knowledge_graph_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="developer",
    maintainer_email="rreisr00@estudiantes.unileon.es",
    description="ROS 2 lifecycle adapter for the knowledge_graph library with SQLite persistence.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "knowledge_graph_bridge = knowledge_graph_ros.knowledge_graph_bridge_node:main",
        ],
    },
)
