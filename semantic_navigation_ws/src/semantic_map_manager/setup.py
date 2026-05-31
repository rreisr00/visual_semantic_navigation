from glob import glob

from setuptools import find_packages, setup

package_name = "semantic_map_manager"

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
    maintainer_email="user@example.com",
    description="Semantic map manager nodes for visual-semantic navigation.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "siglip_inference = semantic_map_manager.siglip_inference:main",
            "waypoint_capture = semantic_map_manager.waypoint_capture:main",
            "semantic_navigator = semantic_map_manager.semantic_navigator:main",
            "test_integration = semantic_map_manager.test_integration:main",
            "visual_encoder = semantic_map_manager.visual_encoder_node:main",
            "kg_manager = semantic_map_manager.kg_manager_node:main",
            "semantic_orchestrator = semantic_map_manager.semantic_orchestrator_node:main",
            "evaluation_node = semantic_map_manager.evaluation_node:main",
        ],
    },
)
