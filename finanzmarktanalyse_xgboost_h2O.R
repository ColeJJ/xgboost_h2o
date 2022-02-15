####### H2O IMPLEMENTIEREN (einmalig nur fuer Installation) ######
# The following two commands remove any previously installed H2O packages for R.
if ("package:h2o" %in% search()) { detach("package:h2o", unload=TRUE) }
if ("h2o" %in% rownames(installed.packages())) { remove.packages("h2o") }

# Next, we download packages that H2O depends on.
pkgs <- c("RCurl","jsonlite")
for (pkg in pkgs) {
  if (! (pkg %in% rownames(installed.packages()))) { install.packages(pkg) }
}

# Now we download, install and initialize the H2O package for R.
install.packages("h2o", type="source", repos="https://h2o-release.s3.amazonaws.com/h2o/rel-zorn/1/R")

####### INIT #######
library(xts) #extensible time series, erweiterung des ts Formats, fuer Finanzzeitreihen
library(quantmod) #funktionen um auf yahoo finance api zuzugreifen
library(h2o)
h2o.init()

path <- "/Users/tounland/Datasets"
setwd(path)

####### 1. FINANZMARKTDATEN LADEN #######
# Zeitintervalle bestimmen, hier kann varriert werden
train_start_date = "2019-01-01"
train_end_date = "2019-12-31"
test_start_date = "2020-01-01"
test_end_date = "2020-12-31"

#Train Daten laden
xetra_data_train <- getSymbols(
  "^GDAXI", src="yahoo", 
  from=train_start_date, 
  to=train_end_date, 
  auto.assign = FALSE
)

#Test Daten laden
xetra_data_test <- getSymbols(
  "^GDAXI", src="yahoo", 
  from=test_start_date, 
  to=test_end_date, 
  auto.assign = FALSE
)

####### 2. FINANZMARKTDATEN AUFBEREITEN #######
closeTrain <- xetra_data_train$GDAXI.Close
closeTest <- xetra_data_test$GDAXI.Close

#Daten skalieren
maxTrain <- max(closeTrain)
minTrain <- min(closeTrain)
maxTest <- max(closeTest)
minTest <- min(closeTest)

closeTrain.scaled <- as.xts(scale(closeTrain, center=minTrain, scale=maxTrain - minTrain))
closeTest.scaled <- as.xts(scale(closeTest, center=minTest, scale=maxTest - minTest))

#DataFrames erstellen
dataTrain <- data.frame (
  z = closeTrain.scaled,
  lag(closeTrain.scaled, 1),
  lag(closeTrain.scaled, 2),
  lag(closeTrain.scaled, 3),
  lag(closeTrain.scaled, 4),
  lag(closeTrain.scaled, 5)
)
dataTrain <- dataTrain [6:nrow(dataTrain), ]
colnames(dataTrain) <- c("z", "t1", "t2", "t3", "t4", "t5")
head(dataTrain)

dataTest <- data.frame (
  z = closeTest.scaled,
  lag(closeTest.scaled, 1),
  lag(closeTest.scaled, 2),
  lag(closeTest.scaled, 3),
  lag(closeTest.scaled, 4),
  lag(closeTest.scaled, 5)
)
dataTest <- dataTest [6:nrow(dataTest), ]
colnames(dataTest) <- c("z", "t1", "t2", "t3", "t4", "t5")
head(dataTest)

# Abspeichern der Models für den späteren Import bei H2O
write.csv(dataTrain, file="xg_train_model.csv")
write.csv(dataTest, file="xg_test_model.csv")

####### 3. DEFINING AND TRAIN AN XGBOOST MODEL #######
##Sources
# https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/xgboost.html

# Import the datasets into H2O:
xg_closeTrain <- h2o.importFile("/Users/tounland/Datasets/models/xg_train_model.csv")
xg_closeTest <- h2o.importFile("/Users/tounland/Datasets/models/xg_test_model.csv")

# we remove C1 and z from predictor columns
predictors <- setdiff(colnames(xg_closeTrain), colnames(xg_closeTrain)[1:2])
# z will be our response / to predicted column
response <- colnames(xg_closeTrain)[2]

# inspect variables: data table + columns + columns
xg_closeTrain
predictors
response 

# Split the dataset into a train and valid set:
xg_closeTrain_splits <- h2o.splitFrame(data =  xg_closeTrain, ratios = 0.8, seed = 1234)
train <- xg_closeTrain_splits[[1]]
valid <- xg_closeTrain_splits[[2]]


# Build and train the model:
closeTrain_xgb <- h2o.xgboost(x = predictors,
                              y = response,
                              training_frame = train,
                              validation_frame = valid,
                              booster = "dart",
                              normalize_type = "tree",
                              ntrees = 100,
                              col_sample_rate_per_tree = 0.9,
                              score_tree_interval = 100,
                              max_depth = 10,
                              learn_rate = 0.02,
                              sample_rate = 0.7,
                              min_rows = 5,
                              seed = 1234)

closeTrain_xgb

####### 4. PREDICTION WITH XGBOOST MODEL #######
#Prediction
pred <- h2o.predict(closeTrain_xgb, newdata = xg_closeTest)
# skalierte Werte zurückberechnen
pred <- pred * (maxTest - minTest) + minTest
pred

# Daten aufbereiten fuer die Visualisierung
dataContainer <- data.frame(
  day=as.Date(character()),
  pred=numeric(),
  ist_value=numeric()
)

for ( i in 1:nrow(dataTest)) {
  current_date = rownames(dataTest[i,])
  prediction = pred[i,1]
  ist_value = closeTest[current_date,1]
  df <- data.frame(current_date, prediction, ist_value)
  dataContainer <- rbind(dataContainer, df)
}

dataContainer

####### 5. VISUALISIERUNG #######
line1 <- dataContainer$prediction
line2 <- dataContainer$GDAXI.Close
plot(line1, ylim=(c(min(line1, line2), max(line1, line2))),
     type="l",
     col="blue",
     main="Vergleich der vorhergesagten und tatsächlichen Close Werten",
     ylab="Dax"
     )
lines(line2, col="red")
legend("topleft",
       c("predict", "ist"),
       cex=0.5,
       lwd=c(1,1),
       col=c("blue", "red"))
