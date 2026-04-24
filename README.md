# NLP： Hidden Markov Model Text Analysis Project

## Overview

This project explores the fascinating world of **Hidden Markov Models (HMMs)**, a powerful mathematical tool that helps us understand patterns in sequences of data. Imagine trying to predict the weather based on what you've observed – HMMs work similarly by learning hidden patterns in visible data.

In this project, we apply HMMs to analyze text data, discovering underlying structures in written language. By training models on text sequences, we can uncover how different "states" (like writing styles or topics) influence the choice of words and letters.

## What Are Hidden Markov Models?

Think of HMMs as detectives that solve mysteries in data:

- **Hidden States**: These are the secret "moods" or "contexts" that influence what we observe
- **Observations**: The visible data we can see (like letters in text)
- **Transitions**: How the hidden states change over time
- **Emissions**: The probability of seeing certain observations from each state

For example, in text analysis, different states might represent different writing styles – one state for formal writing, another for casual conversation.

## Project Goals

This implementation demonstrates:
- How HMMs can learn patterns from text data
- The difference between simple (2-state) and complex (4-state) models
- How machine learning algorithms automatically discover meaningful patterns
- Visualization of learning progress and model interpretations

## Technical Implementation

### Core Components

**HMM Class (`hmm.py`)**:
- Implements the mathematical foundations of Hidden Markov Models
- Uses log-space computations for numerical stability
- Includes forward-backward algorithm for probability calculations
- Implements Baum-Welch algorithm (Expectation-Maximization) for learning

**Main Script (`main.py`)**:
- Processes text data (converts letters and spaces to numerical IDs)
- Initializes HMMs with different numbers of states
- Trains models using 600 iterations of EM algorithm
- Generates comprehensive analysis and visualizations

### Key Features

1. **Text Preprocessing**:
   - Converts text to lowercase
   - Maps 26 letters + space to numerical IDs (0-26)
   - Handles text sequences as observation sequences

2. **Model Training**:
   - 2-state HMM: Simple model with hand-crafted initial parameters
   - 4-state HMM: Complex model with randomized initialization
   - 600 EM iterations to learn optimal parameters

3. **Analysis & Visualization**:
   - Learning curves showing model improvement over time
   - Emission probability evolution for key letters ('a' and 'n')
   - State interpretation through emission pattern analysis
   - Transition matrix analysis

### Data

- **Training Data**: `textA-1.txt` - Used to learn model parameters
- **Test Data**: `textB-1.txt` - Used to evaluate model generalization
- **Alphabet**: 26 English letters + space character

## How to Run

1. Ensure you have Python 3.x installed with NumPy and Matplotlib
2. Place text files in the `data/` directory
3. Run the main script:
   ```bash
   python main.py
   ```

The script will:
- Load and preprocess the text data
- Train both 2-state and 4-state HMMs
- Generate plots in the `plots/` directory
- Print analysis results to console

## Results and Insights

The project generates several key insights through comprehensive visualizations:

### 1. Learning Progress
These plots show how the model's predictive ability improves over 600 training iterations:

**2-State HMM Learning Curve**  
![2-State HMM Log-Probability Curves](plots/2state_Qb_logprob.png)

**4-State HMM Learning Curve**  
![4-State HMM Log-Probability Curves](plots/4state_Qb_logprob.png)

The curves track average log-probability on both training (text A) and test (text B) data, demonstrating how the model learns to better predict text patterns.

### 2. State Discovery Through Emission Evolution
These visualizations reveal how different hidden states specialize in emitting different letters:

**2-State HMM: Emission Probabilities for 'a' and 'n'**  
![2-State HMM Emission Evolution](plots/2state_Qc_emissions_a_n.png)

**4-State HMM: Emission Probabilities for 'a' and 'n'**  
![4-State HMM Emission Evolution](plots/4state_Qc_emissions_a_n.png)

The plots show how emission probabilities for letters 'a' and 'n' evolve across states during training. Horizontal lines indicate the actual frequency of these letters in the training data.

### 3. Model Complexity Comparison
- **2-State Model**: Simple binary classification of text patterns
- **4-State Model**: More complex representation allowing for richer pattern discovery
- **Generalization**: Test performance shows how well learned patterns apply to unseen text

### 4. State Interpretation
The console output provides detailed analysis of:
- Which letters each state prefers to emit
- Transition probabilities between states
- How states differentiate themselves through emission patterns

## Why This Matters

HMMs are fundamental to many modern technologies:
- Speech recognition systems
- Natural language processing
- Bioinformatics (DNA sequence analysis)
- Financial modeling
- Gesture recognition

This project provides a hands-on introduction to these powerful concepts, showing how mathematical models can automatically discover meaningful patterns in real-world data.

## Future Extensions

The framework can be extended to:
- Analyze different languages or writing styles
- Incorporate more complex state structures
- Apply to other sequence data (music, sensor readings, etc.)
- Implement advanced variants like Hierarchical HMMs
