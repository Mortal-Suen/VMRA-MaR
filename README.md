# VMRA-MaR
## Abstract
Breast cancer remains a leading cause of mortality worldwide and is typically detected via screening programs where healthy people are invited in regular intervals. Automated risk prediction approaches have the potential to improve this process by facilitating dynamically screening of high-risk groups. While most models focus solely on the most recent screening, there is growing interest in exploiting temporal information to capture evolving trends in breast tissue, as inspired by clinical practice. Early methods typically relied on two time steps, and although recent efforts have extended this to multiple time steps using Transformer architectures, challenges remain in fully harnessing the rich temporal dynamics inherent in longitudinal imaging data. In this work, we propose to instead leverage Vision Mamba RNN (VMRNN) with a state-space model (SSM) and LSTM-like memory mechanisms to effectively capture nuanced trends in breast tissue evolution. To further enhance our approach, we incorporate an asymmetry module that utilizes a Spatial Asymmetry Detector (SAD) and Longitudinal Asymmetry Tracker (LAT) to identify clinically relevant bilateral differences. This integrated framework demonstrates notable improvements in predicting cancer onset, especially for the more challenging high-density breast cases and achieves superior performance at extended time points (years four and five), highlighting its potential to advance early breast cancer recognition and enable more personalized screening strategies.


![Alt text](scripts/structure.png)

## Requirements

```
cd /path/to/VMRAMaR
pip -r install ./requirements.txt
```

## Training and evaluation

The checkpoint pretrained on Mirai named **mgh_mammo_MIRAI_Base_May20_2019,mgh_mammo_cancer_MIRAI_Transformer_Jan13_2020mgh_mammo_cancer_MIRAI_Transformer_Jan13_2020** to initialize the VMRA-MaR backbone is available [here](https://www.dropbox.com/scl/fi/xdbvk606omyc9yus8mbze/oncoserve_mirai.0.5.0.tar?rlkey=lkuc9aqejfd6dmbj3t8xieqez&e=1&dl=0).

Grid searches:

```
python scripts/dispatcher.py --experiment_config_path configs/vmramar_base.json --result_path vmramar_base_sweep.csv
```

Evaluation:

```
python scripts/dispatcher.py --experiment_config_path configs/vmramar_full.json --result_path vmramar_full_sweep.csv
```
## Acknowledgments
We would like to thank the authors of Yala, whose Onconet GitHub code this work is based upon.
