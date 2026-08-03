from glob import glob

from setuptools import find_packages, setup

package_name = "semantic_evaluation"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/rviz", glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="developer",
    maintainer_email="rreisr00@estudiantes.unileon.es",
    description=(
        "UF-7 evaluation tooling (metric collection + CSV, teleop capture, "
        "operator GUI and knowledge-graph RViz visualizer). Pure-Python core, "
        "no ROS in core/."
    ),
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "evaluation_collector = semantic_evaluation.evaluation_collector:main",
            "graph_visualizer = semantic_evaluation.graph_visualizer:main",
            "semantic_operator_gui = semantic_evaluation.operator_gui:main",
            "teleop_capture = semantic_evaluation.teleop_capture:main",
        ],
    },
)
