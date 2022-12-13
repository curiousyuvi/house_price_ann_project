# Import Required packages
install.packages("neuralnet")
set.seed(500)
library(neuralnet)

# data-set taken from Kaggle
# https://www.kaggle.com/datasets/yashpaloswal/ann-car-sales-price-prediction?select=car_purchasing.csv

# ready data
data <- house_price[,c(-1,-15,-16,-17,-18)]

# Normalize the data
maxs <- apply(data, 2, max)
mins <- apply(data, 2, min)
scaled <- as.data.frame(scale(data, center = mins, scale = maxs - mins))

# Split the data into training and testing set
index <- sample(1:nrow(data), round(0.75 * nrow(data)))
train_ <- scaled[index,]
test_ <- scaled[-index,]

# Build Neural Network
nn <- neuralnet(price~ bedrooms + bathrooms + sqft_living+sqft_lot+floors+waterfront+view+condition+sqft_above+sqft_basement+yr_built+yr_renovated, data = train_, hidden = c(5, 3), linear.output = TRUE)

# Predict on test data
pr.nn <- compute(nn, test_)

# Compute mean squared error
pr.nn_ <- pr.nn$net.result * (max(data$price) - min(data$price)) + min(data$price)
test.r <- (test_$price) * (max(data$price) - min(data$price)) + min(data$price)
MSE.nn <- sum((test.r - pr.nn_)^2) / nrow(test_)

# Plot the neural network
plot(nn)


# Plot regression line
plot(test.r, pr.nn_, col = "red", main = 'Real vs Predicted')
abline(0, 1, lwd = 2)
