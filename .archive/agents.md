# AI Agent Onboarding: Master's Thesis Repository
**Working Title:** "Throughput-Aware Multi-Agent Reinforcement Learning for Dynamic Aerial Relay Swarms in Contested Urban Environments."
**Objective:** Ground-up development of an energy-aware, jam-resilient tactical MANET (Mobile Ad-Hoc Network) UAV swarm.

---

## 1. Core Project Vision & Strategy
We train a homogeneous swarm of tactical autonomous drones to cooperatively track a moving High-Value Target (HVT) convoy in a dense urban environment. The environment is highly contested: direct line-of-sight (LoS) to the Mobile Command Vehicle (MCV) is blocked by concrete structures, and the convoy is equipped with a roof-mounted Counter-UAS Electronic Warfare (EW) jammer. 

The swarm must execute Centralized Training, Decentralized Execution (CTDE) to handle changing configurations natively. To survive, the swarm must optimize a multi-hop relay chain, maintain a 5 Mbps data link back to the MCV, and dynamically rotate high-drain roles (active tracking) and low-drain roles (stationary corner-relaying) based entirely on local Graph Neural Network (GNN) message passing.

---

## 2. Tech Stack & Execution Constraints
To protect a 5-month timeline, all simulations, math operations, and neural network evaluations must execute strictly inside GPU VRAM (`cuda:0`).
*   **Simulator:** NVIDIA Isaac Sim (Imports OpenStreetMap city grids via USD).
*   **Gym Framework:** NVIDIA Isaac Lab (Handles tensorized environment batched stepping).
*   **RL Engine:** skrl (Turnkey Multi-Agent PPO / MAPPO mapped to Isaac Lab tensors).
*   **Graph Framework:** PyTorch Geometric (PyG) (Builds neural nets over dynamic configurations).
*   **BANNED DEPENDENCIES:** Do NOT suggest or import stable-baselines3, Ray/RLlib, OmniDrones, NVIDIA Sionna, SUMO, or NS-3. All channel modeling and physics are computed natively via PyTorch tensors.

---

## 3. Mathematical Foundations & Physics Engines

### A. Joint Energy Model (Kinematic + Telecom)
Drones have a finite battery capacity $B_t \in [0, 1]$. The swarm must optimize a critical trade-off: burn kinematic energy to fly closer to neighbors, or burn telecom energy to transmit louder over long distances. 
The Actor network outputs a 4-dimensional continuous action: `[dv_x, dv_y, dv_z, delta_P_tx]`. 
Power consumption $P_{\text{total}}$ at timestep $t$ combines flight draw and radio transmission draw:

$$P_{\text{total}} = P_{\text{hover}} + \alpha \|\mathbf{v}_t\|^2 + \beta \|\mathbf{a}_t\|^2 + \omega \cdot 10^{(P_{tx} / 10)}$$

*   $P_{\text{hover}}$: Baseline hovering cost.
*   $\alpha, \beta$: Kinematic scaling coefficients (drag, maneuvering).
*   $P_{tx}$: Current transmit power in dBm (clamped between 0 dBm and 30 dBm).
*   $\omega$: RF power amplifier inefficiency scalar.

### B. Geometric Virtual Sensor (The "View")
Do NOT use camera rendering or CNNs. Tracking is evaluated geometrically via a fast virtual sensor:
1.  **Distance Constraint:** Euclidean distance $d$ between the tracker and HVT must be less than $D_{\text{max}}$.
2.  **Field of View (FoV) Cone:** The angle between the drone's negative Z-axis and the vector to the HVT must be within a configured threshold $\theta_{\text{max}}$.
3.  **Occlusion:** An Isaac Sim native PhysX Raycast from the drone to the HVT must return the HVT entity. If it hits a skyscraper mesh, line-of-sight is lost.

### C. 5G Mobile Communications & Electronic Warfare Math
Both friendly and hostile RF signals degrade using the 3GPP TR 38.901 Urban Micro-Cell (UMi) path-loss formulas over a 3.5 GHz carrier band. Concrete building meshes introduce a sharp structural attenuation penalty (e.g., -20 dB) via batched PhysX Raycasting.

1.  **Logarithmic Signal Calculation:**
    $$\text{SINR}_{\text{dB}} = P_{\text{signal}} - (P_{\text{jam}} + N_0)$$
    *   $P_{\text{signal}}$: Link budget from sending node to receiving node.
    *   $P_{\text{jam}}$: Noise floor received from the moving convoy's roof-mounted jammer.
    *   $N_0$: Thermal noise floor (constant at -100 dBm).

2.  **Linear Conversion & Shannon Capacity:**
    $$\text{SINR}_{\text{linear}} = 10^{(\text{SINR}_{\text{dB}} / 10)}$$
    $$\text{Capacity}_{\text{Mbps}} = \frac{B \cdot \log_2(1 + \text{SINR}_{\text{linear}})}{1,000,000}$$
    *   $B$: 5G channel bandwidth (e.g., 20 MHz).

---

## 4. Environment vs. GNN Interface Architecture
To keep reinforcement learning stable, we strictly separate the **binary logical cuts** required by the environment from the **continuous features** passed into the neural network.

### A. PyTorch Geometric (PyG) Continuous Graph Edges
Do NOT use hard binary cutoffs for GNN edges; this creates a gradient cliff. The communication graph uses continuous edge weights $E_{i,j}$ mapped through a temperature-scaled Sigmoid function relative to our application target ($\text{Threshold} = 5.0\text{ Mbps}$):

$$E_{i,j} = \sigma((\text{Capacity}_{\text{Mbps}} - \text{Threshold}) \times \gamma)$$

This continuously scales the features during PyG message passing. If a node is heavily jammed, its incoming edge weight approaches $0.0$, forcing neighbors to implicitly ignore its data without altering the tensor shape.

### B. Environment Evaluation (Isaac Lab Logic)
A discrete application threshold enforces success: `is_link_alive = link_capacity_mbps >= 5.0`.
*   **Episode Termination:** Triggered if `is_link_alive == False` for more than 5 consecutive timesteps, or if *any single drone* hits $B_t = 0$.
*   **Swarm Variance Penalty:** The reward function actively penalizes the variance of the swarm's battery levels: $r_{\text{var}} = -\lambda \cdot \text{Var}(B_1, B_2, \dots, B_N)$. This forces the homogeneous policy to dynamically rotate tracker and relay tasks.

*   **Dynamic Transmission Linkage:** The $P_{tx}$ chosen by the Actor network natively replaces the static `tx_power_friendly` in the 3GPP path-loss calculations at every timestep. This directly alters the resulting $\text{SINR}_{\text{dB}}$ and immediately feeds back into the continuous GNN edge weight calculations.

---

## 5. Coding Conventions for the AI Agent
*   **Device Isolation:** Never call `.cpu()` or `.numpy()` inside the main simulation loop. All environment tensors must live and stay on `cuda:0`.
*   **File Proximity:** Place test scripts directly inside the directory of the module they validate (e.g., `src/comms/test_channel.py` sits right next to `src/comms/channel.py`).
*   **Nomenclature:** Use tactical/telecom terms consistently: `hvt_convoy`, `mcv_base`, `tactical_node`, `sinr_db`, `shannon_mbps`, `edge_weight`.
*   **Data Structure:** Node observations are 13-dimensional vectors passing local kinematics, battery level, target metrics, and ambient noise floor.