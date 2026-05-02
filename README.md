This project focuses on the automated segmentation of brain tumors from multi-modal MRI scans using a state-of-the-art hybrid deep learning architecture. By combining local feature extraction with global contextual awareness, the system achieves a high level of accuracy and reliability suitable for medical research environments

Technical Implementation & Architecture
Hybrid U-Net Design: The model, titled AttentionCNNTransformerUNet, utilizes a standard encoder-decoder structure but enhances the bottleneck with a Transformer layer to capture long-range dependencies across the image.
Spatial Attention: You implemented Attention Gates within the skip-connections of the U-Net. These gates help the model focus on the most relevant features by suppressing activations in non-tumor regions before they are passed to the decoder.
Multi-Modal Input: The pipeline is designed to ingest four distinct MRI modalities: FLAIR, T1, T1ce, and T2, providing the model with a comprehensive view of different tissue characteristics.
Data Strategy: The BraTSDataset class includes a specific logic to load only tumor-containing slices, which optimizes the training process by focusing on the most informative data points.
Advanced Loss Function: To combat the high class imbalance between healthy tissue and tumor, you used a hybrid Dice + Cross-Entropy Loss (DiceCELoss) with a heavy weight (0.9) assigned to the tumor class.

Performance Metrics
The model was rigorously trained for 30 epochs using the AdamW optimizer. The following results were achieved during evaluation
Training Convergence: The model reached a Final Train Loss of 0.0735, showing stable and effective learning throughout the training phase.
Standard Accuracy: Initial patient-level testing resulted in a Dice WT score of 0.9009, demonstrating strong overlap between predicted segments and ground truth.
Uncertainty-Aware Accuracy: When employing uncertainty estimation, the accuracy slightly improved to a Dice WT of 0.9014
High Confidence: The average Prediction Uncertainty was calculated at 0.004405, indicating that the model is highly confident in its segmentation boundaries.

Reliability & Uncertainty Estimation
A standout feature of this project is the inclusion of Monte Carlo (MC) Dropout for clinical reliability.
Stochastic Inference: By enabling dropout during the evaluation phase and performing 10 forward passes per slice, the model generates a distribution of predictions rather than a single static output.
Entropy-Based Measurement: You calculated the entropy of these predictions to provide a pixel-wise and patient-level uncertainty score.
Clinical Value: This approach allows a medical professional to see not just where the model thinks a tumor is, but also how sure it is about that specific area, which is a critical requirement for high-stakes medical AI applications

Project Structure & Reproducibility
The repository is organized for ease of use and professional deployment
Core Logic: model_2.py houses the architecture, while dataset_2.py and losses_2.py manage data and optimization.
Evaluation Suite: Separate scripts for slice-level evaluation (eval_2.py), patient-level evaluation (eval_patient_2.py), and uncertainty analysis (eval_patient_uncertainty_2.py) allow for multi-faceted testing.
Clean Workflow: The .gitignore_2 file ensures that heavy datasets and local environment files are not uploaded, maintaining a lightweight and professional repository footprint.
