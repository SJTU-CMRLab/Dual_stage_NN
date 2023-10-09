# A Dual-Stage Partially Interpretable Neural Network for Joint Suppression of bSSFP Banding and Flow Artifacts in Non-Phase-Cycled Cine Imaging

# Abstract

**Purpose**

To develop a partially interpretable neural network for joint suppression of banding and flow artifacts in non-phase-cycled bSSFP cine imaging


<picture>
 <source media="(prefers-color-scheme: dark)" srcset="https://github.com/SJTU-CMRLab/Dual_stage_NN/blob/main/Figure1.png">
 <source media="(prefers-color-scheme: light)" srcset="https://github.com/SJTU-CMRLab/Dual_stage_NN/blob/main/Figure1.png">
 <img alt="The scheme of the dual-stage network and generation of the training labels" src="https://github.com/SJTU-CMRLab/Dual_stage_NN/blob/main/Figure1.png">
</picture>


**Methods**

A dual-stage neural network consisting of a Voxel-Identification (VI) sub-network and Artifact-Suppression (AS) sub-network is proposed. The VI sub-network provides identification of artifacts, which guides artifact suppression and improves interpretability. The AS sub-network reduces banding and flow artifacts. 

Short-axis cine images of 12 frequency offsets from 28 healthy subjects were used to train and test the dual-stage network. Additional 77 patients were retrospectively enrolled to evaluate its clinical generalizability. For healthy subjects, artifact suppression performance was analyzed by comparison with traditional phase cycling. The partial interpretability provided by the VI sub-network was analyzed via correlation analysis. Generalizability was evaluated for cine obtained with different sequence parameters and scanners. For patients, artifact suppression performance and partial interpretability of the network were qualitatively evaluated by 3 clinicians. Cardiac function before and after artifact suppression was assessed via Left Ventricular Ejection Fraction (LVEF).

**Results**

For the healthy subjects, visual inspection and quantitative analysis found a considerable reduction of banding and flow artifacts by the proposed network. Compared with traditional phase cycling, the proposed network improved flow artifact scores (4.57±0.23 vs 3.40±0.38, P=0.002) and overall image quality (4.33±0.22 vs 3.60±0.38, P=0.002). The VI sub-network well identified the location of banding and flow artifacts in the original movie and significantly correlated with the change of signal intensities in these regions. Changes of imaging parameters or the scanner did not cause a significant change of overall image quality relative to the baseline dataset, suggesting a good generalizability. For the patients, qualitative analysis showed a significant improvement of banding artifacts (4.01±0.50 vs 2.77±0.40, P<0.001), flow artifacts (4.22±0.38 vs 2.97±0.57, P<0.001), and image quality (3.91±0.45 vs 2.60±0.43, P<0.001) relative to the original cine. The artifact suppression slightly reduced the LVEF (mean bias=-1.25%, P=0.01).

**Conclusions**

The dual-stage network simultaneously reduces banding and flow artifacts in bSSFP cine imaging with a partial interpretability, sparing the need for sequence modification. The method can be easily deployed in a clinical setting to identify artifacts and improve cine image quality.

# Installation



# How to use
Details of the codes are as follows:

[VIstage_train.py](https://github.com/SJTU-CMRLab/Dual_stage_NN/blob/main/VIstage_train.py): To train VI stage firstly.

[ASstage_train.py](https://github.com/SJTU-CMRLab/Dual_stage_NN/blob/main/ASstage_train.py): To train AS stage after training VI stage. During the training of AS stage, VI sub-network was frozen and only used to provide the VI maps.

[test.py](https://github.com/SJTU-CMRLab/Dual_stage_NN/blob/main/test.py): To test the dual-stage network.

[data](https://github.com/SJTU-CMRLab/Dual_stage_NN/blob/main/data): It contains training data and testing data. Cine images with 12 frequency offsets are included in the training data.

[model](https://github.com/SJTU-CMRLab/Dual_stage_NN/blob/main/model): It contains trained VI and AS models.

