# Federated Intrusion Detection for Connected Energy Devices

These are benchmarks for distributed (federated) AI that detects network attacks across many sites without pooling raw data. They are the completed first stage of an independent project on secure AI monitoring for connected solar and grid-edge systems, by Khandoker Samiul Hoque (ORCID 0009-0008-0756-2395).

## Contents
| Folder | Study | Key result |
|---|---|---|
| `02_Federated_IoT_IDS/` | FedAvg on NSL-KDD with 10 simulated clients, against centralized comparators | The federated model approaches centralized performance while raw data stays local |
| `poisoning_study/` | Five-seed model-poisoning study (one attacker sends reversed updates) | Under attack, accuracy is **0.251** for FedAvg and **0.401** for clipped FedAvg, while **coordinate median holds 0.757**, matching the no-attack baseline of **0.755** |

## Reproduce
1. Download NSL-KDD from the University of New Brunswick (https://www.unb.ca/cic/datasets/nsl.html) or its public mirror.
2. Place `NSL-KDD-Dataset-master.zip` in `data/`.
3. Run:
```bash
pip install -r requirements.txt
python 02_Federated_IoT_IDS/code/analysis.py
cd poisoning_study && python run_fedids_extension_experiment.py
```
**Reruns on 2026-10-15:** both studies matched the saved results. The baseline tables differed by at most 2e-8. The poisoning summary was identical.

## Limits
- **Historical data.** NSL-KDD is a historical benchmark, not current energy-network traffic.
- **Simulated clients.** The clients are simulated.
- **Privacy.** No formal differential-privacy guarantee or secure aggregation is claimed. Keeping data local is not the same as formal privacy.
- **Scope of the poisoning result.** Coordinate median resisted the one attack tested. This does not establish general robustness.

## Next stages
- Modern IoT and energy-network datasets.
- Implemented and tested privacy mechanisms.
- Multiple adaptive attacks.

License: MIT (code). NSL-KDD is not redistributed here.
