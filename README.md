AI-Based Drug–Drug Interaction Prediction and Drug Combination Recommendation System

Overview

This project presents an Artificial Intelligence (AI)-based Drug–Drug Interaction (DDI) Prediction and Drug Combination Recommendation System developed as part of a master's thesis. The system employs a Graph Attention Network (GAT) to predict the severity of drug–drug interactions and recommend potential drug combinations based on pharmacological knowledge.
The proposed framework is designed to support research into antiepileptic drugs; however, it can be adapted to other therapeutic areas by replacing the input datasets with disease-specific data.
The model integrates multiple heterogeneous biomedical data sources, including:
-Known Drug–Drug Interactions (DrugBank-style dataset)
-Drug mechanisms of action
-Drug molecular targets
-Disease subtype–drug prescription relationships
-Graph-based representation learning using PyTorch Geometric
The objective is to predict unreported DDIs and recommends potentially safe drug combinations and flag risky combinations.
The proposed Graph Attention Network (GAT)-based drug interaction prediction model achieved an overall accuracy of 64.44% and a weighted F1-score of 65.00% on a held-out test set of DrugBank interactions. Considering the relatively small dataset (42 drugs and 494 labeled interactions) and the multi-class prediction task (mild, moderate, and severe interactions), these results demonstrate that the model successfully captures meaningful biological and pharmacological relationships. In addition to interaction prediction, the proposed framework integrates biological targets, mechanism-of-action embeddings, disease subtype information, and shared-target safety verification to provide biologically informed drug combination recommendations.

---------------------------------------------------------------------------------------------------------------------------------------------

System Features

The proposed framework provides the following capabilities:
-Graph Attention Network (GAT)-based link prediction
-Multi-class DDI severity prediction: Mild, Moderate, Severe
-Integration of heterogeneous biomedical features
-Molecular targets
-Mechanism of action embeddings
-Disease subtype information
-Automatic identification of high-risk combinations through shared biological targets
-Construction of a biomedical knowledge graph
-Prediction of unknown drug interactions
-Interactive drug recommendation interface
-Knowledge graph visualization
-Performance evaluation using standard machine learning metrics

---------------------------------------------------------------------------------------------------------------------------------------------

Model workflow
                   Install Required Libraries
                               │
                               ▼
                   Upload Required CSV Files
                               │
                               ▼
                       Load All Datasets
                               │
                               ▼
                     Data Preprocessing
       • Normalize drug names
       • Remove incomplete/invalid records
       • Encode interaction severity labels
       • Parse biological targets
       • Build drug–target mapping
                               │
                               ▼
              Biological Feature Engineering
      ┌──────────────────────────────────────────────┐
      │                                              │
      │ Target Features                              │
      │ (Binary Drug–Target Matrix)                  │
      │                                              │
      ├──────────────────────────────────────────────┤
      │                                              │
      │ Mechanism of Action Embeddings               │
      │ (SentenceTransformer MiniLM-L6-v2)           │
      │                                              │
      ├──────────────────────────────────────────────┤
      │                                              │
      │ Disease Subtype Features                     │
      │ (Drug–Disease Prescription Relationships)    │
      │                                              │
      └──────────────────────────────────────────────┘
                               │
                               ▼
              Feature Fusion & Normalization
       • Concatenate all feature matrices
       • L2 feature normalization:
	Normalize each drug feature vector to unit length.
	Balance heterogeneous biological features before graph learning.
	Improve numerical stability and training convergence.
                               │
                               ▼
            Drug Interaction Graph Construction
       • Nodes  = Approved drugs
       • Edges  = Known DrugBank DDIs
       • Labels = Mild / Moderate / Severe
                               │
                               ▼
                Dataset Partitioning
       • Training interaction graph
       • Held-out evaluation interactions
                               │
                               ▼
              Graph Attention Network (GAT)
       • Four GAT layers
       • Batch normalization
       • Residual aggregation of multi-layer node embeddings
       • Edge representation learning
       • Multi-layer classifier (MLP)
                               │
                               ▼
                    Model Optimization
       • AdamW optimizer
       • Weighted Cross-Entropy Loss
       • Learning-rate scheduler
       • Early stopping
                               │
                               ▼
                 Performance Evaluation
       • Accuracy
       • Weighted F1-score
       • Classification report
                               │
                               ▼
          Prediction of Unknown Drug Interactions
       • Generate unseen drug pairs
       • Predict interaction severity
         (Mild / Moderate / Severe)
                               │
                               ▼
        Biological Validation of Predictions
       • Compare shared molecular targets
       • Flag biologically high-risk combinations
                               │
                               ▼
         Drug Recommendation & Decision Support
       • Retrieve known interactions
       • Display predicted interactions
       • Recommend safer combinations
       • Identify risky combinations
       • Display mechanism of action
       • Display disease subtype indications
                               │
                               ▼
            Interactive Knowledge Graph
       • Complete heterogeneous biomedical graph
       • Drug-specific interaction network
       • Color-coded severity visualization


---------------------------------------------------------------------------------------------------------------------------------------------
Graph Attention Network (GAT) workflow 
                           

			Input Node Features
      (Target Features + MoA Embeddings + Disease Subtypes)
                                   │
                                   ▼
                        Drug Interaction Graph
                   Nodes = Drugs | Edges = Known DDIs
                                   │
                                   ▼
                     ┌──────────────────────────┐
                     │      GAT Layer 1         │
                     │ GATConv (4 heads)        │
                     │ BatchNorm + ReLU         │
                     └──────────────────────────┘
                                   │
                                   ▼
                     ┌──────────────────────────┐
                     │      GAT Layer 2         │
                     │ GATConv (4 heads)        │
                     │ BatchNorm + ReLU         │
                     └──────────────────────────┘
                                   │
                                   ▼
                     ┌──────────────────────────┐
                     │      GAT Layer 3         │
                     │ GATConv (4 heads)        │
                     │ BatchNorm + ReLU         │
                     └──────────────────────────┘
                                   │
                                   ▼
                     ┌──────────────────────────┐
                     │      GAT Layer 4         │
                     │ GATConv (4 heads)        │
                     └──────────────────────────┘
                                   │
                                   ▼
                  Residual Feature Aggregation
                 h = h₁ + h₂ + h₃ + h₄
                                   │
                                   ▼
                    Edge Representation Learning
          ┌────────────────────────────────────────────┐
          │ Source Node Embedding (src)                │
          │ Destination Node Embedding (dst)           │
          │ |src − dst|                                │
          │ src × dst                                  │
          └────────────────────────────────────────────┘
                                   │
                                   ▼
                Concatenate Edge Feature Vector
                                   │
                                   ▼
                     Multi-Layer Perceptron (MLP)
             Linear(1024→128) → ReLU → Dropout(0.3)
                                   │
                                   ▼
                          Linear(128→64) → ReLU
                    	           │
                                   ▼
                             Dropout(0.2)
                                   │
                                   ▼
                         Linear(64→3 Classes)
                                   │
                                   ▼
                   Predicted Interaction Severity
                      Mild | Moderate | Severe

Architecture of the proposed Graph Attention Network (GAT) for drug–drug interaction prediction. Drug node representations are generated from heterogeneous biological features, including molecular targets, mechanism-of-action embeddings, and disease subtype information. Four graph attention layers learn node representations, whose outputs are aggregated through residual connections. Pairwise edge representations are constructed by combining the source and destination node embeddings and are classified by a multi-layer perceptron into three interaction severity levels (mild, moderate, and severe).

In our model, each drug is represented as a node, and each node has a feature vector obtained by concatenating three different biological feature sets. These node features are what the GAT layers use to learn meaningful representations of drugs. 
Node Features or feature vector of each drug node consists of:
1. Biological Target Features (Binary Encoding)
These features indicate the molecular targets of each drug. The source of this feature is mechanism_of_action - target.csv input file, and represented in binary vector (0/1). Each dimension corresponds to one biological target. Value = 1 if the drug acts on that target, otherwise 0.
This allows the GAT to learn similarities between drugs acting on common molecular targets.
2. Mechanism of Action Semantic Embedding
Instead of using only text, the mechanism of action is converted into a numerical embedding using SentenceTransformer (all-MiniLM-L6-v2)
Example: Mechanism "Noncompetitive NMDA receptor antagonist“ 384-dimensional embedding vector[0.024, -0.182, ..., 0.067]. These embeddings capture the semantic meaning of pharmacological mechanisms, allowing drugs with similar mechanisms to have similar representations even if the wording differs.
3. Disease Subtype Prescription Features
These features describe whether a drug is prescribed for specific disease subtypes. The Source of this feature is disease_edges.csv input file, and represented in binary vector.
This provides disease-specific clinical information that helps the model recommend combinations within relevant therapeutic contexts.
Final Node Feature Vector
The final feature vector is created by concatenating all three feature groups: 
𝑋_𝐷𝑟𝑢𝑔=[𝑇𝑎𝑟𝑔𝑒𝑡 𝐹𝑒𝑎𝑡𝑢𝑟𝑒𝑠∣∣𝑀𝑒𝑐ℎ𝑎𝑛𝑖𝑠𝑚 𝐸𝑚𝑏𝑒𝑑𝑑𝑖𝑛𝑔∣∣𝐷𝑖𝑠𝑒𝑎𝑠𝑒 𝑆𝑢𝑏𝑡𝑦𝑝𝑒 𝐹𝑒𝑎𝑡𝑢𝑟𝑒𝑠] ​
After concatenation, L2 normalization is applied. This scales each drug feature vector to unit length, preventing one feature type from dominating the learning process.
Both the weight matrix and the attention scores are learned automatically during training, but they depend on different things.
1. Weight Matrix (W)
In each GAT layer, the first operation is a linear transformation.
PyTorch Geometric automatically creates this matrix. 𝑊 is learned from the training data. During every epoch, until the prediction error becomes as small as possible. Specifically, it is optimized using:
Drug target features 
Mechanism embeddings 
Disease subtype features 
Known DrugBank interaction labels 
Cross-entropy loss 
Backpropagation 
AdamW optimizer
Each of the four GAT layers learns its own weight matrix. 
2. Attention Score
After transforming node features, GAT computes how important each neighboring drug is by comparing their targets, mechanisms of action,  and disease subtype usage.
3. Softmax Normalization
These scores are normalized and all neighbors' coefficients sum to 1.
Meaning that more informative neighboring drugs have a greater influence.
The attention score depends on the transformed node features, which originate from:
Target similarity (shared biological targets) 
Mechanism of action embeddings 
Disease subtype prescription features 
Graph connectivity (known DDIs) 
Thus, if two drugs have similar biological and clinical characteristics, the GAT can learn to assign them higher attention weights.
Multi-Head Attention
The model has multi-head attention which means that four independent attention mechanisms are learned in parallel. Each head has its own weight matrix 𝑊 and attention vector 𝑎, allowing the model to capture different aspects of the drug relationships—for example, one head may focus more on target similarity, another on mechanism similarity, and another on disease subtype information. The outputs of the four heads are averaged rather than concatenated, producing a stable hidden representation of size 256 for each drug.

---------------------------------------------------------------------------------------------------------------------------------------------
Input Data

The framework requires the following CSV files:
ddi_edges.csv
drug_nodes.csv
disease_nodes.csv
disease_edges.csv
mechanism_of_action - target.csv

The epilepsy datasets included in this repository are provided as an example implementation.

---------------------------------------------------------------------------------------------------------------------------------------------
Overall Training Workflow


Known Drug Features + Known DDIs
              │
              ▼
       Graph Attention Network
              │
              ▼
 Predict Interaction Severity
 (Mild / Moderate / Severe)
              │
              ▼
 Compare Prediction with
 Actual DrugBank Label
              │
              ▼
     Cross-Entropy Loss
 (Measures Prediction Error)
              │
              ▼
      Backpropagation
 (Calculates How Parameters
     Should Be Updated)
              │
              ▼
      AdamW Optimizer
 (Updates Model Weights)
              │
              ▼
 Updated GAT Model
              │
              ▼
 Repeat for Next Epoch


1. Cross-Entropy Loss
Cross-Entropy Loss is the function that measures how wrong the model's prediction is. It compares the actual interaction severity with the predicted interaction severity.
The smaller the loss, the better the prediction.

2. Backpropagation
Backpropagation is the learning process. It detects which model parameters caused the prediction error, and how should they change. 
It computes the gradient (direction of improvement) for every learnable parameter; Weight matrices (W), Attention vectors (a), MLP classifier weights, and Batch normalization parameters, using the chain rule of calculus.

3. AdamW Optimizer
The optimizer updates the model parameters using the gradients calculated during backpropagation. Its goal is to reduce the loss over time.
AdamW improves on the standard Adam optimizer by using decoupled weight decay, which helps reduce overfitting and often improves generalization.
It is especially effective for graph neural networks, transformer models, and high-dimensional feature representations.
---------------------------------------------------------------------------------------------------------------------------------------------

Required Dataset Structure:

1. ddi_edges.csv
Required columns:
Drug1
Drug2
Severity
Interaction (optional)
Severity values must be one of:
mild
moderate
severe

2. drug_nodes.csv
Required columns:
Drug

3. disease_nodes.csv
Required columns:
Disease_Subtype
grouped disease subtypes

4. disease_edges.csv
Required columns:
Drug
Disease_Subtype
Prescribed
Note (optional)
The Prescribed column should contain:
yes
no

5. mechanism_of_action - target.csv
Required columns:
Drug
Mechanism
Target
Multiple targets should be separated using "/".
Example:
Sodium channel/GABA receptor

---------------------------------------------------------------------------------------------------------------------------------------------

Feature Engineering

Unlike traditional graph models that use only node identifiers, this framework constructs biologically meaningful node representations by combining three feature types:

1. Molecular Target Features
Binary feature vectors indicating the biological targets associated with each drug.

2. Mechanism of Action Embeddings
Mechanism descriptions are converted into dense semantic embeddings using the Sentence Transformer model:all-MiniLM-L6-v2

3. Disease Subtype Features
Binary vectors indicating whether a drug is prescribed for specific disease subtypes.

The final node feature matrix is obtained by concatenating:
-Target Features
-Mechanism Embeddings
-Disease Subtype Features
followed by L2 normalization.

---------------------------------------------------------------------------------------------------------------------------------------------

Model Architecture

The proposed model is implemented using PyTorch Geometric.
Architecture:
-Four Graph Attention Network (GAT) layers
-Batch Normalization
-Residual feature aggregation
-Multi-layer edge classifier

Edge representations are generated by concatenating:
-Source node embedding
-Destination node embedding
-Absolute embedding difference
-Element-wise embedding product

The classifier predicts one of three interaction classes:
-Mild
-Moderate
-Severe

---------------------------------------------------------------------------------------------------------------------------------------------

Training Strategy

The model incorporates several techniques to improve robustness and generalization:
-Stratified train/test split
-Held-out evaluation dataset
-AdamW optimizer
-Weighted Cross-Entropy Loss
-Learning rate scheduling (ReduceLROnPlateau)
-Early stopping
-Best model checkpoint saving and loading
-Class imbalance handling using class weights

---------------------------------------------------------------------------------------------------------------------------------------------

Knowledge Graph Construction

The generated knowledge graph contains:
Nodes:
-Drug nodes
-Disease subtype nodes
-Disease category nodes
Edges:
-Drug–Drug Interactions
-Drug–Disease prescriptions
-Disease hierarchy relationships

---------------------------------------------------------------------------------------------------------------------------------------------

Prediction Workflow

After training, the framework:
-Predicts missing drug–drug interactions.
-Estimates interaction severity.
-Identifies shared biological targets.
-Recommends potentially safe drug combinations.
-Flags biologically risky combinations.

---------------------------------------------------------------------------------------------------------------------------------------------

Interactive Recommendation Interface

The notebook provides an interactive interface using ipywidgets.
The user can:
-Select any drug from the dataset
-View prescribed disease subtypes
-Display mechanism of action
-Retrieve known interactions
-View predicted interactions
-Receive safe combination recommendations
-Receive risky combination warnings
-Visualize the interaction network

---------------------------------------------------------------------------------------------------------------------------------------------

Visualization

Two graph visualizations are generated.

1. Complete Biomedical Knowledge Graph
Includes:
-Drugs
-Disease subtypes
-Disease groups
-Prescription relationships
-Predicted DDIs
Node Colors:
Node-Color
Drug-Blue
Disease Subtype-Green
Disease Group-Coral
Edge Colors:
Relationship-Color
Prescribed-Green
Not Prescribed-Red
Subtype-of-Gray
Mild DDI-Blue
Moderate DDI-Orange
Severe DDI-Red

2. Drug-Specific Interaction Graph
Displays:
-Known DDIs
-Predicted DDIs
-Severity labels
-Safe and risky combinations

---------------------------------------------------------------------------------------------------------------------------------------------

Model Evaluation

Performance is evaluated using:
-Accuracy
-Weighted F1-score
-Precision
-Recall
-Classification Report
A held-out dataset is used for unbiased evaluation of prediction performance.

---------------------------------------------------------------------------------------------------------------------------------------------

Software Requirements

Recommended environment:
Python 3.10
PyTorch 2.1.2
PyTorch Geometric 2.5.3
Sentence Transformers 2.7.0
Transformers 4.41.2
Scikit-learn 1.5.0
NumPy 1.26.4
Pandas 2.2.2
NetworkX 3.3
Matplotlib 3.9.0
ipywidgets 8.1.3

Install dependencies using:
pip install -r requirements.txt

or execute the installation cells in the provided Google Colab notebook.

---------------------------------------------------------------------------------------------------------------------------------------------

How to Run

-Install the required packages.
-Launch the notebook.
-Upload the required CSV datasets.
-Execute all notebook cells.
-Select a drug from the dropdown menu.
-Click Run Prediction.
-Review the generated recommendations and visualizations.

---------------------------------------------------------------------------------------------------------------------------------------------

Model Output 

Disease 2 drug name
Disease 2 subtype(s) that the drug is prescribed for
Prescription status (Yes or No)
Mechanism of Action (MoA)
Known or predicted DDI, severity, and interaction note

---------------------------------------------------------------------------------------------------------------------------------------------

Important Notes

-Shared biological targets automatically increase the predicted interaction risk.
-Predictions are intended to support biomedical research and hypothesis generation.
-The framework is not intended for clinical decision-making.
-Prediction quality depends heavily on the completeness and accuracy of the input datasets.

---------------------------------------------------------------------------------------------------------------------------------------------

Future Work

Possible extensions include:
-Integration with DrugBank or other biomedical databases through APIs
-Dosage-aware interaction prediction
-Incorporation of protein–protein interaction (PPI) networks
-Inclusion of pharmacokinetic and pharmacodynamic features
-Explainable AI techniques for interpreting GNN predictions
-External validation using independent clinical datasets
-Multi-disease and multi-drug recommendation support

---------------------------------------------------------------------------------------------------------------------------------------------

License

This software is provided for research and educational purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or clinical decision-making.