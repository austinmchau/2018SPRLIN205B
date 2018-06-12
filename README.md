# Linguistics 205B
Austin Chau
## Introduction
The goal of this project is to investigate the effect of dataset sizes on the performance of different language models. Every language model requires training data to learn, and the quality of the model depends on the quality of the data. However, while scraping data nowadays may be easy due to the advent of the Internet, these data are often noisy and unsanitized. Finding labeled data for supervised learning often requires manual labor to tag these data. 

While the general consensus is that having more data means better model, this theory is often left untested across models. Perhaps one model may outperform another despite having a smaller dataset. In recent years, most natural language processors are often implemented using neural networks, where linguistics model built upon rules and understanding of the language are less considered for use. However, these linguistics model may outperform neural networks when the dataset is small.

This project compares the performance between a multiclass perceptron – representing a standard linear language model – and an LSTM neural network using an intent classification problem. By feeding the models with different sizes of dataset and measuring their accuracy, the difference in performance can be found.

## Instructions
### Prerequisites
This project requires Python 3.6 and additional packages listed in the requirements.txt . The codes are self-documented, describing its function as necessary. Using virtualenv is recommended. 

Download the DSTC2 dataset from [here](http://camdial.org/~mh521/dstc/). Place the file in the `data/` directory, such that the folder should look like:
```
/root
| data
| | dstc2_test
| | dstc2_traindev
``` 
The Keras dataset should download on its own, otherwise, follow the instructions from Keras on getting the dataset.

###  Running the code
The entry point of the code is evaluate.py . 
```bash
python3 evaluate.py
```
The code is not set up for taking arguments from the command line currently, but the parameters can be changed in code as described in the script.

This script runs both models on both datasets – DSTC2 and the Keras Reuters classification dataset. The script will run each model with four iteration/epoch settings with the same dataset sizes. The result will then be printed out.

*Note: Currently, the LSTM model will fail on the Reuters dataset when the dataset size is 0.1 and 0.2 of the original. This is due to the dataset being too small and does not contain all the labels in the testing set.* 

## Presentation
The slides can be found [here](https://docs.google.com/presentation/d/15-K2xzoMKHhroBUoTO1XjPy7ZxUGU3EDAj02kVOjaj0/edit?usp=sharing). Requires UC Davis GMail account.

## Results
The script will run all both models (perceptron, lstm) on both datasets (dstc2, reuters) for dataset sizes from 0.1 to 1.0 of the original incrementing by 0.1. This is repeated with four iteration/epoch settings.

The results can be found in results.xlsx . 

### Analysis
| | |
|:---:|:---|
| ld vs. pd | The perceptron performs better across all data sizes than LSTM. | 
| lr vs. pr | The LSTM performs better across all data sizes (> 0.3) than the Perceptron. |

####Plot of LSTM vs. Perceptron on DSTC2
![ld_pd_chart](/results/ld_pd.png?raw=true "Plot of LSTM vs. Perceptron on DSTC2")
####Plot of LSTM vs. Perceptron on Reuters
![ld_pd_chart](/results/lr_pr.png?raw=true "Plot of LSTM vs. Perceptron on Reuters")

It is interesting to see the model performs oppositely on the two dataset. This suggest that other than dataset sizes, other factors such as the nature of the dataset itself can affect the performance of the model. The DSTM2 model is a corpus of restaurant search phone calls, whereas the Reuters corpus are news articles. The utterances in DSTC2 are often very similar and the vocabulary is more limited than the Reuters corpus, which contains more vocabulary and more varied sentence structures. DSTC2 has a  smaller set of label as well, whereas the Reuters article has 46 categories. This gives the perceptron an edge on the DSTC2 as the simpler model may be able to pick up more of the underlying structures of the dataset, but it will do worse on the Reuters data due to the larger perplexity. This is the opposite for LSTM, as the model may not have enough variation in the training data to learn on the DSTC2, but enough on the Reuters dataset.

It is also notable that the perceptron's performance on the Reuters is better than the LSTM's performance on the DSTC2. And the LSTM's performance on the Reuters is worse than the Perceptron's performance on the DSTC2. This can be an indication that despite the overall low performance, the perceptron is able to perform better on noisy or limited datasets. The perceptron is can also perform more consistently with varying dataset sizes than the LSTM model.

The perceptron model is also much easier to implement due to its simplicity and with fewer variation in its setup parameters. The results shows that the perceptron model can perform comparably with the LSTM model when the dataset is noisy or limited. This supports the notion that simpler language model is useful in situations where development time is contrained and dataset quality is not good or consistent.

## Conclusion
While the comparison with only two datasets may not concretely determine whether datset sizes have a direct effect on the model performance, the results demonstrates that the models perform better under conditions that are the strong suite of the model. The perceptron, with its simplicity, performs comparibly with the LSTM model under certain conditions. Given that the perceptron model is simpler mathematically and easier to implement that the LSTM model, this shows that simpler language models are still vital in situations where the datset is limited or noisy, or when development time is contrained. 