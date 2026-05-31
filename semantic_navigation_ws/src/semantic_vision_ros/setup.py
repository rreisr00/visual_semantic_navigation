from glob import glob

from setuptools import find_packages, setup

package_name = "semantic_vision_ros"

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
    description="ROS 2 lifecycle node wrapping semantic_vision_core.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "visual_encoder = semantic_vision_ros.visual_encoder_node:main",
        ],
    },
)
