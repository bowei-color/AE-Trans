# AE-Trans
`AE-Trans` (AE-Transformer) is a dual-modality fusion interpretable deep neural network model based on the Transformer architecture, capable of integrating RNA and DNA methylation data for diagnosing Alzheimer's disease and identifying related biomarkers.The model incorporates an autoencoder, a Transformer model, and the integrated gradients attribution method to compute feature contributions, providing accurate and interpretable predictions for Alzheimer's disease diagnosis.

![Fig1 Image](Fig1.png "Figure 1: Description of the image")

## Features

- **High Accuracy**: The model demonstrated outstanding performance on the test set, with an accuracy of 92.63% and an AUC of 96.76%.
- **Interpretability**: Model interpretation is performed using integrated gradients attribution, allowing the identification of biomarkers related to Alzheimer's disease.
- **Dual-modality Fusion**: Leveraging the Transformer to effectively integrate the underlying complementary information between RNA and DNA methylation data, generating better latent representations of both RNA and DNA methylation data.
- **Dimensionality Reduction**: Using the AE to effectively reduce data dimensionality and extract important information from the features.

## Getting Started

Ensure you have the following prerequisites installed:
- Python >= 3.9.x
- PyTorch == 2.5.0 
- NumPy == 1.24.3
- Pandas == 2.2.3
- Scikit-learn == 1.5.1

You can install the required packages using `pip`:

```bash
pip install torch numpy scikit-learn
```

or use the following command to install all dependencies：

```bash
pip install -r requirements.txt
```


### Installation

To use the `AE-Trans` model, clone the repository:

```bash
git clone https://github.com/bowei-color/AE-Trans.git
```

Create a virtual environment using conda:

```bash
conda create --name environment_for_aetrans python=3.9
```

Activate the virtual environment

```bash
conda activate environment_for_aetrans
```

Navigate to the project directory

```bash
cd AE-Trans/model
```

### Usage

To run the model with the provided dataset, execute the following command:

```python
python AE-Trans.py --epochs[number of iterations] --file_path_gene[your gene data file path] --file_path_methy[your methylation data file path]
```

## Input structure

AE-Trans implements a binary classification task, where the input includes RNA data and DNA methylation data. In both the RNA and DNA methylation data, the first column contains the sample names, the second column contains the labels, and the features start from the third column onwards.

### Example for data:

```bash
  Sample      Label Feature1  Feature2   Feature3 ...
GSM2139163      1    0.235     1.378      3.546
GSM2139164      0    10.731    0.556      6.034
GSM2139165      0    3.265     2.931      2.878
GSM2139166      1    5.307     4.251      11.215
```

The datasets can be accessed at https://doi.org/10.5281/zenodo.13640497
        
        .

## Documentation
The project documentation provides a detailed description of the model, the logic of the algorithm, an explanation of the parameters, and a guide on how to reproduce the experimental results. Here is a summary of the documentation:

- **Model Architecture**: AE-Trans is a bimodal fusion interpretable deep network model based on the Transformer architecture.
- **Algorithm Logic**: The model employs the integrated gradients method to provide interpretability of decisions, helping to identify key features that affect predictions.
- **Parameter Explanation**: A detailed list of all parameters involved in the model training and prediction processes and their functions.

## Results
Our model has been evaluated using five-fold cross-validation and the test set. Below is a summary of some key results:

- **Accuracy**: The model achieved an average accuracy of 92.63% and an average AUC of 96.76% on the human Alzheimer's disease dataset.
- **Performance Metrics**: In addition to accuracy, the model also performed well in precision, recall, and F1 score.
- **Model Comparison**: Compared to existing methods, AE-Trans has shown significant improvements across various evaluation metrics.

## Contributing

Contributions to the project are welcome. Please submit a pull request or create an issue in the GitHub repository.

## Contact Authors

Dr. Bowei Yan: boweiyan2020@gmail.com
