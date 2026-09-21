# Architecture

Observation history → graph forecaster → multi-step prediction + uncertainty → decentralized MAPPO actors → signal actions → simulator. Training uses centralized-critic MAPPO conceptually; V1 keeps the policy interface small so the synthetic pipeline is testable without a GPU. The next implementation step is a PyTorch actor/critic and SUMO TraCI adapter behind the existing interfaces, followed by a 10–30 junction Bangalore-inspired graph.

