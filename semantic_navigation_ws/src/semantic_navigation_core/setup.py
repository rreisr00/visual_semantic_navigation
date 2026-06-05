from setuptools import find_packages, setup

package_name = "semantic_navigation_core"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="developer",
    maintainer_email="rreisr00@estudiantes.unileon.es",
    description="Pure-Python semantic navigation logic (ranking + capture state machine). No ROS 2 dependency.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={},
)
