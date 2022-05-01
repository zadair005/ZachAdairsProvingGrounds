###Linear Regression Example
##Linear Regression is used to predict the value of an outcome variable Y based 
# on 1+ input predictor variables X. The aim is to establish a linear relationship 
# (a mathematical formula) between the predictor variable(s) & the response variable, 
# so that, we can use this formula to estimate the value of the response Y, when only 
# the predictors (Xs) values are known. 

##Introduction
# The aim of linear regression is to model a continuous variable Y as a mathematical function
# of 1+ X variable(s), so that we can use this regression model to predict the Y when only the X
# is known. This mathematical equation can be generalized Y = B(1) + B(2)X + Epislon where, B(1) 
# is the intercept and B(2) is the slope. Collectively, they're called regression coefficients
# (epislon) is the error term, the part of Y the regression model is unable to explain.

##Example Data
#For this analysis, we will use cars dataset:
data(cars)
head(cars)

#Before we begin building the regression model, it is a good practice to analyze and understand the 
# variables. THe graphical analysis & correlation study below will help w/ this.

##Graphical Analysis
The aim of this exercise is to build a simple regression modle that we can use to predict Distance (dist)
by establishing a statistically significant linear regression w/ speed. Before though, lets try to 
understand these variables graphically. For each of the independent variables (predictors), the following 
plots are drawn to visualize the following behavior.
1) Scatter plot: Visualize the linear regression between the predictor & response.
2) Box plot: To spot any outlier observations in the variable. Having outliers in your predictor can drastically
affect the predictions as they can easily affect the direction/slope of the line of best fit. 
3) Density Plot: To see the distribution of the predictor variable. Ideally, a close to normal distribution 
(Bell shaped curve), w/out being skewed to the left or right is preferred. Let us see how to make each one 

#Scatter plot
Scatter plots can help visualize any linear relationships between the dependent and independent variables. 
Ideally, if you have multiple predictor variables, a scatter plot is drawn for each of them against the response, 
along w/ the line of best fit.

scatter.smooth(x=cars$speed, y=cars$dist, main = "Dist - Speed")

The scatter plot along w/ the smoothing line above suggests a linearly increasing relationship between the 2 
variabls. This is good, cause 1 of the underlying assumptions in linear regression is that the relationship between 
the response & predictor variables is linear & additive.

#Box plot - Check for outliers
Any datapoint that lines outside (1.5*IQR) is considered an outlier. 

par(mfrow=c(1,2)) #divide the graph into 2 columns
boxplot(cars$speed, main= "Speed", sub=paste("Outlier rows: ", boxplot.stats(cars$speed)$out)) #Boxplot for speed
boxplot(cars$dist, main="Distance", sub=paste("Outlier rows ", boxplot.stats(cars$dist)$out)) #boxplot for Distance

#Density Plots - Check if the response variable is close to normality
library(e1071)
par(mfrow=c(1,2))
plot(density(cars$speed), main = "Density Plot: Speed", ylab="Frequency", sub=paste("Skewness:", round(e1071::skewness(cars$speed), 2))) #Density plot for 'speed'
polygon(density(cars$speed), col='red')
plot(density(cars$dist), main="Density Plot: Distance", ylab = "Frequency", sub=paste("Skewness:", round(e1071::skewness(cars$dist), 2))) # Density Plot for 'Distance'
polygon(density(cars$dist), col="red")

#Correlation
Correlation is a statistical measure that suggests the level of linear dependence between 2 variables, that occur in 
pair - just like what we have here in speed & dist. Correlation can take values between -1 to +1. If we observe for 
every instance where speed increases, the distance also increases along w/ it, then there is a high positive correlation
between them & therefore the correlation between them will be closer to 1. The opposite is true for an inverse relationship, 
in which case, the correlation between the variables will be close to -1.

A value closer to 0 suggests a weak relationship between the variables. A low correlation (-0.2 < x 0.2)
probably suggests that much of variation of the response variable (Y) is unexplained by the predictor (X), in which case, we 
should probably look for better explanatory variables.

cor(cars$speed, cars$dist) #calculate correlation between speed & distance

#Build Linear Model
Now that we have seen the linear relationship pictorally in the scatter plot and by computing the correlation, lets see the 
syntax for building the linear model. THe function used for building linear models is lm(). The lm() function takes in 2 main 
arguments, namely 1. Formula 2. Data. THe data is typically a data.frame and the formula is a object of class formula. But the 
most common convention is to write out the formula directly in place of the argument as written below.

linearMod <- lm(dist ~ speed, data=cars) #Build linear model regression model on full data
print(linearMod)

Now that we have built the linear model, we also have established the relationship between the predictor & response in th eform of a 
mathematical formula for Distance (dist) as a function for speed. For the above output, you can notice the 'Coefficients' part having 
2 components: Intercept -17.579, speed: 3.932. These are also called the beta coefficients. 
In other words, dist = Intercept (B + speed) => dist = -17.579 + 3.932*speed

#Linear Regression Diagnostics
Now the linear model is built and we have a formula we can use to predict the dist value if a corresponding speed is known. Is this 
enough to actually use this model? NO! Before using a regression model, you have to ensure that it is statistically significant. How do 
you ensure this? Lets begin by printing the summary stats for linearMod

summary(linearMod) #model summary

#The p-value: Check for statistical significance 
The summary stats above tells us a number of things. One of them is the model p-value (bottom last line) and the p-value of individual 
predictor variables (extreme right column under 'Coefficients'). The p-values are very important because We can consider a linear model
to be statistically significant only when both these p-values are less that the pre-determined statistical significance level, which is ideally 0.05.
This is visually interpreted by the significance stars at the end of the row. The more the stars besides the variable's p-value, the more significant 
the variable.

Null and Alternate Hypothesis
When there is a p-value, there is a null & alternative hypothesis associated with it. In Linear Regression, the Null Hypothesis is that the coefficients
associated w/ the variables is equal to zero. The alternate hypothesis is that the coefficients are not equal to zero (i.e. there exists a relationship 
between the independent variable in question & the dependnet variable). 

t-value
We can interpret the t-value something like this. A larger t-value indicates that it is less likely that the coefficient is not equal to zero purely by chance.
So, higher the t-value, the better. Pr(>|t|) or p-value is the probability that you get a t-value as higher or higher than the observed value when the Null Hypothesis 
(the B coefficient is equal to zero or that there is no relationship) is true. So if the Pr(>|t|) or p-value is the probability that you get a t-value as high or higher 
than the observed value when the Null Hypothesis (the B coefficient is equal to zero or that there is no relationship) is true. So if the Pr(>|t|) is low, the coefficients
are significant (significantly different from zero). If the Pr(>|t|) is high, the coefficients are not signficant.

What this means to us? When p-value is less than significance level (< 0.05), we can safely reject the null hypothesis that the co-efficient B of the predictor is zero. In our case,
linearMod, both these p-Values are well below the 0.05 threshold, so we can conclude our model is indeed statistically significant.

It is absolutely important for the model to be statistically significant before we can go ahead & use it to predict (or estimate) the dependent variable, otherwise,
the confidence in predicted values from that model reduces and may be constructed as an event of chance.

When the model co-efficients and standard error are known, the formula for calculating t Statistic & p-value is as follows:
t-statistic = B - coefficient / Std. Error"

modelSummary <- summary(linearMod) #capture model summary as an object
modelCoeffs <- modelSummary$coefficients #model coefficients
beta.estimate <- modelCoeffs["speed", "Std. Error"] #get std. error for speed
std.error <- modelCoeffs["speed", "Std. Error"] #get std.error for speed
t_value <- beta.estimate/std.error # calc t-statistic
p_value <- 2*pt(-abs(t_value), df=nrow(cars)-ncol(cars)) #calc p-value
f_statistic <- linearMod$fstatistic[1] #fstatistic
f <- summary(linearMod)$fstatistic # parameter for model p-value calc 
model_p <- pf(f[1], f[2], f[3], lower=FALSE)
modelSummary

#R-Squared and Adj R-Squared
The actual info in a data is the total variation it contains. What R-Squared telss us is the proportion of variation
in the dependent variable that has been explained by this model.
R squared = 1 - SSE/SST where, SSE is the Sum of Square Errors
We don't necessarily discard a model based on a low R-Squared value. Its a better practice to look at the AIC and prediction accuracy on validation sample when 
deciding on the efficacy of a model.

Now thats about R-Squared. What about adjusted R-Squared? As you add more X variables to your model, the R-Squared value of the new bigger model will always be
greater than that of the smaller subset. This is cause, since all the variables in the original model is also present, their contribution to explain the dependent 
variable will be present in the super-set as well, therefore, whatever new variable we add can only add (if not significantly) to the variation that was already 
explained. It is here, the adjusted R-Squared value comes to help Adj R-Squared penalizes total value for the number of terms (read predictors) in your model.
Therefore when comparing nested models, it is a good practice to look at adj R-Squared value over R-squared 
R-Squared adjusted = 1 - MSE/MST where, MSE is the mean squared error given by MSE where n is the number of observations and q is the number of coefficients in the
model. Therefore, by moving around the numerators & denominators, the relationship between R squared and R adj squared.

#Standard Error and F-Statistic
Both standard errors and F-statistic are measures of goodness of fit.
Std. Error = Square Root of MSE  = Square root of SSE/n-q
F-Statistic = MSR/MSE where, n is the # of observations, q is the number of coefficients and MSR is the mean square regression.

#AIC and BIC 
The Akaikes Information Criterion - AIC (Akaike, 1974) and the Bayesian information criterion - BIC are measures of the goodness of fit of an estimated statisical model.
and can also be used for model selection. Both criteria depend on the maxed value of the likelihood function L for the estimated model.

The AIC is defined as: 
AIC = (-2) * In(L) + (2*k) where, k is the number of model parameters & the BIC is defined as:
BIC = (-2) * In(L) + k * ln(n) where, n is the sample size. 

For model comparison:
AIC(linearMod) # AIC => 419.1569
BIC(linearMod) # BIC => 424.8929

#How to know if the model is best fit for your data?
The most common metrics to look at while selecting the model are:
R-Squared - Higher the better (>0.7)
Adj R-Squared - Higher the better
F - Statistic - Higher the better
Std. Error - Closer to zero the better
t-statistic - Should be greater than 1.96 for p-value to be less than 0.05
AIC - Lower the better
BIC - Lower the better
Mallows cp - Should be close to the nubmer of predictors in model
MAPE (Mean Absolute Error) - Lower the better
MSE (Mean Squared Error) - Lower the better
Min_Max Accuracy => mean(min(actual, predicted)/max(actual, predicted) - Higher the better

#Predicting Linear Models
So far we have seen how to build a linear regression model using the whole dataset. If we build it that way,
there is no way to tell how the model will perform w/ new data. SO ther preferred practice is to split your dataset into 
a 80/20 sample (training/test) then build the model on the 80% sample and then use the model thus built to predict the 
dependent variable on test data.

Doing it this way, we will have the model predicted values for the 20% (test) as well as the actuals (from the original dataset). 
By calculating accuracy measures (like min_max accuracy) and error rates (MAPE or MSE), we can find out the prediction accuracy 
of the model. Now, lets see how to actually do this.

Step 1: Create the training (development) and test (validation) data samples from original data:
#Create Training & Test data
set.seed(100) #setting seed to reproduce results of random sampling
trainingRowIndex <- sample(1:nrow(cars), 0.8*nrow(cars)) #Rows indices for training data
trainingData <- cars[trainingRowIndex, ] #model train data
testData <- cars[-trainingRowIndex, ] #test data

Step 2: Develop the model on the training data and use it to predict the distance on test data
#Build the model on train data
lmMod <- lm(dist ~ speed, data = trainingData) #Build the model
distPred <- predict(lmMod, testData) #predict distance

Step 3: Review diagnostic measures
summary(lmMod) #model Summary

From the model summary, the model p-value & predictors p value are less than the significance level, so we know we have a
significant model. Also, the R-Square & Adj R-Square are comparative to the original model built on full data.

Step 4: Calculate prediction accuracy and error rates
A simple correlation between the actuals and predicted values can be used as a form of accuracy measure. A higher correlation
accuracy implies that the actuals & predicted values have similar directional movement, i.e. when the actuals values increase 
the predicteds also increase & vice-versa.

actuals_preds <- data.frame(cbind(actuals=testData$dist, predicteds=distPred)) #Make actual predicteds dataframe
correlation_accuracy <- cor(actuals_preds) #82.7
correlation_accuracy
head(actuals_preds)

Now lets calculate the Min/Max accuracy and MAPE:
MinMaxAccuracy = mean(min(actuals, predicteds)/max(actuals,predicteds))
MeanAbsolutePercentageError (MAPE) = mean(abs(predicteds-actuals)/actuals)

min_max_accuracy <- mean(apply(actuals_preds, 1, min) / apply(actuals_preds, 1, max))
mape <- mean(abs((actuals_preds$predicteds - actuals_preds$actuals))/actuals_preds$actuals)

#k-Fold Cross Validation
Suppose, the model predicts satisfactorily on the 20% split (test data), is that enough to believe that your model will perform equally 
well all the time? It is important to rigorously test the models performance as much as possible. One way is to ensure that the model 
performance as much as possible. One way is to ensure that the model equation you have will perform well, when it is 'built' on a different 
subset of training data and predicted on the remaining data. 

How to do this is? Split your data into 'k' mutually exclusive random sample portions, Keeping each portion as test data,
we build the model on the remaining (k-1 portion) data and calculate the mean squared error of the predictions. This is done for each of 
the k random sample portions. Then finally, the average of these MSEs (for k portions) is computed. We can use this metric to compare different 
linear models. 

By doing this, we need to check a few things:
  1. If the models prediction accuracy isnt varying too much for any one particular sample, and 
  2. If the lines of best fit do not vary too much w/ respect to the slope & level
  

In other words, they should be parallel and as close to each other as possible. 
par(mfrow=c(1,1))
library(DAAG)
cvResults <- suppressWarnings(CVlm(cars, form.lm=dist ~ speed, m=5, 
                                   dots=FALSE, seed=29, legend.pos = "topleft", 
                                   printit=FALSE, main = "Small symbols are predicted values while bigger ones are actuals."));  #performs the CV
attr(cvResults, 'ms') #=> 251.2783 MSE

