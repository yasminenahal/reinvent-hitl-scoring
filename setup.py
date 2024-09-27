import setuptools

setuptools.setup(
    name="reinvent_hitl_scoring",
    version="0.0.5",
    author="Yasmine Nahal",
    author_email="yasminenahal@aalto.fi",
    description="Scoring functions for Reinvent adapted to the HITL_AL_GOMG workflow",
    url="https://github.com/yasminenahal/reinvent-hitl-scoring.git",
    package_data={"reinvent_scoring": ["scoring/score_components/synthetic_accessibility/fpscores.pkl.gz"]},
    packages=setuptools.find_packages(exclude='unittest_reinvent'),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
