# The Steam Engine: A Recommendation System for Steam Users (CIS 520 Final Project)
## About
Steam is a digital distribution platform in which users buy and play computer games.  We employ neighborhood, matrix factorization, and mixed collaborative filtering (CF) methods to predict the number of hours users will play games. In addition, we adapt a regression boosting frame-work for matrix factorization CF algorithms and apply it to the prediction task. We find that factorization methods outperform neighborhood methods, and a mixed approach outperforms both. Additionally, we find the boosting framework did not meaningful improve performance. To improve predictions, future research should incorporate user friendship networks.

## Team Members
* Brandon Lin
* Chris Painter
* Barry Plunkett
* Stephanie Shi

## Setting up the Project
### Installing the Dependencies
`pip install -r requirements.txt`

## File Directory
* `neighborhood` - memory-based methods
* `factorization` -  latent factor models
* `boost` - boosting
* `ensemble` - mixed methods

Data is too large to include and can be obtained by contacting the owner.
