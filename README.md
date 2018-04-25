# The Steam Engine: A Recommendation System for Steam Users (CIS 520 Final Project)
## About
Steam is a video game distribution platform. We employ neighborhood, matrix factorization, and mixed collaborative filtering (CF) methods to predict the number of hours Steam users will play games. We also adapt a regression boosting framework for matrix factorization CF algorithms and apply it to the prediction task. We find that neighborhood methods outperform matrix factorization methods, and a mixed approach outperforms both. Additionally, we find the boosting framework did not meaningful improve performance. To improve predictions, future research should incorporate user friendship networks.  

## Team Members
* Brandon Lin
* Chris Painter
* Barry Plunkett
* Stephanie Shi

## File Directory
* `neighborhood` - memory-based methods
* `factorization` -  latent factor models
* `boost` - boosting
* `ensemble` - mixed methods

## Setting up the Project
### Installing the Dependencies
`pip install -r requirements.txt`

Full original dataset can be found [here](https://steam.internet.byu.edu/) and is over 200GB. Processed data is too large to include and can be obtained by contacting the owner.
