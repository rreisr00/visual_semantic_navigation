from glob import glob

from setuptools import find_packages, setup

package_name = "semantic_navigation_ros"

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
    description="Thin ROS 2 coordinators for semantic navigation (capture + retrieval).",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "kg_manager = semantic_navigation_ros.kg_manager_node:main",
            "semantic_orchestrator = semantic_navigation_ros.semantic_orchestrator_node:main",
            "lifecycle_manager = semantic_navigation_ros.lifecycle_manager_node:main",
        ],
    },
)
