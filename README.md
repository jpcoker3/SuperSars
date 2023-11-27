# SuperSars
## Intro
This repository was used for the Graduate Research Project in Introduction to Radar. Our project researches the connection between noise and SAR target classification. We utilized the MSTAR dataset for this project and wrote all code in MATLAB. 

## Data Overview
### Noisy vs Clean Data            

![Noisey Data](https://github.com/jpcoker3/SuperSars/blob/main/Data/Noise/Gaussian/2S1/HB14931.PNG)  ![Clean Data](https://github.com/jpcoker3/SuperSars/blob/main/Data/Padded_imgs/2S1/HB14931.JPG)


## Conclusion
Our research into improving SAR target classification through artificial noise has given us many valuable insights. The original goal was to make our models more reliable in real-world scenarios by training models on noisy data along with clean data. However, our findings revealed several unexpected challenges.  

We tested a variety of models, primarily utilizing the MSTAR dataset, and found that the initial clean model performed extremely well with an accuracy of 99.97%. In comparison, the model trained on noisy data showed an inferior accuracy of 92.67%. This drop in performance indicates that the added noise may have caused the models to struggle to adapt to the new, noisy data.  

Further experiments were conducted where we validated models on noisy data and a variety of noise levels. These experiments highlighted the difficulty of balancing noise for improved performance without hindering accuracy. The models trained on the noisy data were consistently lower than the clean data. This demonstrates the complexity of adding noise without harming the overall model performance.  

 

Our findings show a complex relationship between noise, model performance, and dataset characteristics within SAR target classification. The quick overtraining of the models on noisy data, as well as the lower performance when compared to clean data, suggest the need for a more sophisticated and mathematical approach.  

Future work may include expanding the datasets used to include more diverse and challenging data to analyze performance and validate our initial hypothesis. Exploring alternative noise addition methods, various model architectures, and strategies to combat overfitting could allow us to deepen our understanding and improve the quality of SAR classification models.  

While our attempts to boost SAR target classification accuracy showed potential, the challenges presented demonstrate the need for a more sophisticated approach. With a more sophisticated approach, we may be able to achieve an improved SAR target classification in varying real-world conditions.  
