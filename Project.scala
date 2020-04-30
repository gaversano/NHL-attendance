// Databricks notebook source
import com.mongodb.spark._

// COMMAND ----------

//Read colelction nhl from mongoDB database test 
val nhl_mongo = spark.read.format("com.mongodb.spark.sql.DefaultSource").option("database", "test").option("collection", "nhl").load()

// COMMAND ----------

display(nhl_mongo)

// COMMAND ----------

//Average attendence by team (home, visitors)
display(nhl_mongo)

// COMMAND ----------

//Display maximum attendance by the home team. 
display(nhl_mongo.groupBy($"teams.home").max("attendence"))

// COMMAND ----------

//Investigate why the maximum is much larger for few teams. These games were special events at larger venues and are removed from the dataset.
//The hisogram below shows the attendance is grouped around 21,000 with an outlier aaround 70,000.
val blackhawks = nhl_mongo.filter($"teams.home" === "Chicago Blackhawks")
display(blackhawks.select($"attendence"))

// COMMAND ----------

//import org.apache.spark.sql.functions.{stddev_samp, stddev_pop} 
//display(nhl_mongo.groupBy($"teams.home").agg(stddev_pop($"attendence")))

// COMMAND ----------

//Remove outliers where the attendance is greater than 25,000.
val nhl_mongo_no_outliers = nhl_mongo.filter($"attendence" < 25000)
nhl_mongo_no_outliers.count()

// COMMAND ----------

//Now the ditribution of max attendance is much closer.
display(nhl_mongo_no_outliers.groupBy($"teams.home").max("attendence"))

// COMMAND ----------

//The hisotgram for the Blackhawks after filtering the very large outlier. 
val blackhawks_no_outliers = nhl_mongo_no_outliers.filter($"teams.home" === "Chicago Blackhawks")
display(blackhawks_no_outliers.select($"attendence"))

// COMMAND ----------

//Read tabular data "game_data.txt"
val dataPath = "/FileStore/tables/game_data.txt"
val nhl = sqlContext.read.format("com.databricks.spark.csv")
  .option("header","true")
  .option("inferSchema", "true")
  .load(dataPath)

display(nhl)

// COMMAND ----------

display(nhl.describe())

// COMMAND ----------

//Remove outliters where attendance > 25,000
val nhl_no_outliers = nhl.filter($"attendance" < 25000)

// COMMAND ----------

//Only 14 rows were removed. The mean is very similar but the max and std are lower. 
display(nhl_no_outliers.describe())

// COMMAND ----------

//Group rows by season and home team, take the average attendance 
val nhl_attendance_by_season = nhl_no_outliers.groupBy("season", "home").avg("attendance")

// COMMAND ----------

display(nhl_attendance_by_season)

// COMMAND ----------

display(nhl_no_outliers.groupBy("season").avg("attendance").sort("avg(attendance)"))

// COMMAND ----------

//Top 5 2019 attendance
display(nhl_attendance_by_season.filter($"season" === 2019).sort($"avg(attendance)".desc).take(5))

// COMMAND ----------

//Bottom 5 2019 attendance
display(nhl_attendance_by_season.filter($"season" === 2019).sort($"avg(attendance)").take(5))

// COMMAND ----------

//Blackhawks average attendace by season
display(nhl_attendance_by_season.filter("home == 'Chicago Blackhawks'").sort($"season"))

// COMMAND ----------

//Canadiens average attendace by season
display(nhl_attendance_by_season.filter("home == 'Montreal Canadiens'").sort($"season"))

// COMMAND ----------

//Leafs average attendace by season
display(nhl_attendance_by_season.filter("home == 'Toronto Maple Leafs'").sort($"season"))

// COMMAND ----------

//Flyers average attendace by season
display(nhl_attendance_by_season.filter("home == 'Philadelphia Flyers'").sort($"season"))

// COMMAND ----------

//Red Wings average attendace by season
display(nhl_attendance_by_season.filter("home == 'Detroit Red Wings'").sort($"season"))

// COMMAND ----------

//Devils average attendace by season
display(nhl_attendance_by_season.filter("home == 'New Jersey Devils'").sort($"season"))

// COMMAND ----------

//Senators average attendace by season
display(nhl_attendance_by_season.filter("home == 'Ottawa Senators'").sort($"season"))

// COMMAND ----------

//Coyotes average attendace by season
display(nhl_attendance_by_season.filter("home == 'Arizona Coyotes'").sort($"season"))

// COMMAND ----------

//Panthers average attendace by season
display(nhl_attendance_by_season.filter("home == 'Florida Panthers'").sort($"season"))

// COMMAND ----------

//Islanders average attendace by season
display(nhl_attendance_by_season.filter("home == 'New York Islanders'").sort($"season"))

// COMMAND ----------

//Average attendace by season all teams
nhl_attendance_by_season.show(9999)

// COMMAND ----------

//Attendance standard deviation by team
import org.apache.spark.sql.functions.{stddev_samp, stddev_pop} 
display(nhl_no_outliers.groupBy($"home").agg(stddev_pop($"attendance")))

// COMMAND ----------

//Attendance standard deviation by team histogram
display(nhl_no_outliers.groupBy($"home").agg(stddev_pop($"attendance")))

// COMMAND ----------

//Attendance standard deviation by team by season
display(nhl_no_outliers.groupBy($"home",$"season").agg(stddev_pop($"attendance")).sort($"season"))

// COMMAND ----------

//Attendance standard deviation by team by season histogram
display(nhl_no_outliers.groupBy($"home",$"season").agg(stddev_pop($"attendance")).sort($"season"))

// COMMAND ----------

//Confirmation that ~ 70 % of teams have an attendance standard deviation below 1000. 
val total = nhl_no_outliers.groupBy($"home",$"season").agg(stddev_pop($"attendance")).sort($"season").count()

val small_std = nhl_no_outliers.groupBy($"home",$"season").agg(stddev_pop($"attendance")).sort($"season").filter($"stddev_pop(attendance)" < 1000).count()

// COMMAND ----------

display(nhl_no_outliers)

// COMMAND ----------

//Below is a failed attempt to run through each row of nhl_no_outliers and look up the average attendance given the season and home team to compare to the actual attendance. I was able to write and test the function get_avg_attendance but wa unsuccessful at using it within the map function.


/*
def add_goals (home : Int)(visitor : Int) = home + visitor
val test = add_goals(2)(3)
display(nhl_no_outliers.map(row => add_goals(row.getAs[Int](4))(row.getAs[Int](7))))

def get_avg_attendance (season : Int)(home_team : String) = nhl_attendance_by_season.filter($"season" === season).filter($"home" === home_team).select($"avg(attendance)")

val test2 = get_avg_attendance(2019)("Buffalo Sabres")
display(test2)

display(nhl_no_outliers.map(row => get_avg_attendance(row.getAs[Int](5))(row.getAs[String](3))))
*/

// COMMAND ----------

display(nhl_attendance_by_season)

// COMMAND ----------

//The purpose of the next blok of code is to create a table for a machine learning algorithm to predict the average attendance given the average attendance for the last three years. i.e., features = [2015, 2016, 2017] and output = [2018] and features = [2016, 2017, 2018], output = [2019] 

//The model can be used to predict 2020 average attendance where features = [2017, 2018, 2019], output = [2020] 

import org.apache.spark.sql.functions._ 

val attendance_2015 = nhl_attendance_by_season.filter("season = 2015").select($"home", $"avg(attendance)".alias("2015"))
val attendance_2016 = nhl_attendance_by_season.filter("season = 2016").select($"home", $"avg(attendance)".alias("2016"))
val attendance_2017 = nhl_attendance_by_season.filter("season = 2017").select($"home", $"avg(attendance)".alias("2017"))
val attendance_2018 = nhl_attendance_by_season.filter("season = 2018").select($"home", $"avg(attendance)".alias("2018"))
val attendance_2019 = nhl_attendance_by_season.filter("season = 2019").select($"home", $"avg(attendance)".alias("2019"))


// COMMAND ----------

val joined1 = attendance_2015.join(attendance_2016, Seq("home")).join(attendance_2017, Seq("home")).join(attendance_2018, Seq("home"))
//display(joined)

val joined2 = attendance_2016.join(attendance_2017, Seq("home")).join(attendance_2018, Seq("home")).join(attendance_2019, Seq("home"))
//display(joined2)

val joined_final = joined1.union(joined2).withColumnRenamed("2015", "year1").withColumnRenamed("2016", "year2").withColumnRenamed("2017", "year3").withColumnRenamed("2018", "year4")
display(joined_final)

// COMMAND ----------

//Transform data into input vector and output column for use in ML

import org.apache.spark.ml.feature.VectorAssembler

val assembler = new VectorAssembler()
  .setInputCols(Array("year1", "year2", "year3"))
  .setOutputCol("features")

val nhlFinal = assembler.transform(joined_final)
display(nhlFinal)

// COMMAND ----------

//Split into training and testing sets

val Array(training, test) = nhlFinal.randomSplit(Array(0.7, 0.3))

// Going to cache the data to make sure things stay snappy!
training.cache()
test.cache()

println(training.count())
println(test.count())

// COMMAND ----------

//Create linear regression model using features as the input and price as the output

import org.apache.spark.ml.regression.LinearRegression

val lrModel = new LinearRegression()
  .setLabelCol("year4")
  .setFeaturesCol("features")
  .setElasticNetParam(0.5)

println("Printing out the model Parameters:")
println("-"*20)
println(lrModel.explainParams)
println("-"*20)

// COMMAND ----------

//Fit the training set

import org.apache.spark.mllib.evaluation.RegressionMetrics
val lrFitted = lrModel.fit(training)

// COMMAND ----------

//Check the values of the coefficients and intercept
//weight of coefficients is larger for more recent years. However, the weights for year t-3 and t-2 are negative. 
println(lrFitted.coefficients)
println(lrFitted.intercept)

// COMMAND ----------

//Transform the test data using the fitted model
val holdout = lrFitted
  .transform(test)
  .selectExpr("prediction", 
    "year4")
display(holdout)

// COMMAND ----------

//Look at regression metrics

val rm = new RegressionMetrics(
  holdout.select("prediction", "year4").rdd.map(x =>
  (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double])))

println("MSE: " + rm.meanSquaredError)
println("MAE: " + rm.meanAbsoluteError)
println("RMSE Squared: " + rm.rootMeanSquaredError)
println("R Squared: " + rm.r2)
println("Explained Variance: " + rm.explainedVariance + "\n")

// COMMAND ----------

display(holdout.withColumnRenamed("year4", "actual"))

// COMMAND ----------

//Use the model to predict 2020 average attendance using 2017, 2018 and 2019 average attendance. 

val joined3 = attendance_2017.join(attendance_2018, Seq("home")).join(attendance_2019, Seq("home")).withColumnRenamed("2017", "year1").withColumnRenamed("2018", "year2").withColumnRenamed("2019", "year3")
display(joined3)

val nhl2020 = assembler.transform(joined3)
display(nhlFinal)


val predicted_2020 = lrFitted
  .transform(nhl2020)

display(predicted_2020)

// COMMAND ----------



// COMMAND ----------

display(nhl_no_outliers)

// COMMAND ----------

import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.{Pipeline, PipelineStage}


//Convert the string categories into numeric indicies

val featureCol = Array("home", "visitor")
val indexers = featureCol.map { colName =>
  new StringIndexer().setInputCol(colName).setOutputCol(colName + "Index")
}

val pipeline = new Pipeline().setStages(indexers)      
val nhl_no_outliers_Index = pipeline.fit(nhl_no_outliers).transform(nhl_no_outliers)

display(nhl_no_outliers_Index)

// COMMAND ----------

import org.apache.spark.ml.feature.OneHotEncoderEstimator


//Convert the numeric indices into a binary vector. 
//Note that these categories are ordinal and could perhaps be encoded into an integer [worst, best]

val encoder = new OneHotEncoderEstimator()
  .setInputCols(Array("homeIndex", "visitorIndex"))
  .setOutputCols(Array("homeCat", "visitorCat"))

val nhl_no_outliers_Cat = encoder.fit(nhl_no_outliers_Index).transform(nhl_no_outliers_Index)
nhl_no_outliers_Cat.show()

// COMMAND ----------

//Transform data into input vector and output column for use in ML

import org.apache.spark.ml.feature.VectorAssembler

val assembler = new VectorAssembler()
  .setInputCols(Array("homeCat", "visitorCat"))
  .setOutputCol("features")

val nhl_no_outliers_w_feats = assembler.transform(nhl_no_outliers_Cat)
display(nhl_no_outliers_w_feats)

// COMMAND ----------

val assembler2 = new VectorAssembler()
  .setInputCols(Array("attendance"))
  .setOutputCol("output")

val nhl_no_outliers_w_feats_w_output = assembler2.transform(nhl_no_outliers_w_feats)
display(nhl_no_outliers_w_feats_w_output)

// COMMAND ----------

//Scale the attendance to have an std of 1
import org.apache.spark.ml.feature.StandardScaler
val scaler = new StandardScaler()
  .setInputCol("output")
  .setOutputCol("output_scaled")
  .setWithStd(true)
  .setWithMean(false)

val scalerModel = scaler.fit(nhl_no_outliers_w_feats_w_output)


val nhl_no_outliers_scaled = scalerModel.transform(nhl_no_outliers_w_feats_w_output)

nhl_no_outliers_scaled.show()

// COMMAND ----------

//Convert the scaled attendance back to a double for use in LR
//val nhl_no_outliers_final = nhl_no_outliers_scaled

// COMMAND ----------

//Use the attendance rather than the normalzied attendance 
val lrModel2 = new LinearRegression()
  .setLabelCol("attendance")
  .setFeaturesCol("features")
  .setElasticNetParam(0.5)

println("Printing out the model Parameters:")
println("-"*20)
println(lrModel2.explainParams)
println("-"*20)

//Split into training and testing sets

val Array(training2, test2) = nhl_no_outliers_scaled.randomSplit(Array(0.7, 0.3))

// Going to cache the data to make sure things stay snappy!
training2.cache()
test2.cache()

println(training2.count())
println(test2.count())


val lrFitted2 = lrModel2.fit(training2)
val holdout2 = lrFitted2
  .transform(test2)
  .selectExpr("home","visitor","prediction", "attendance")
display(holdout2)

// COMMAND ----------

println(training2.count())
println(test2.count())

// COMMAND ----------

//Look at regression metrics

val rm2 = new RegressionMetrics(
  holdout2.select("prediction", "attendance").rdd.map(x =>
  (x(0).asInstanceOf[Double], x(1).asInstanceOf[Int])))

println("MSE: " + rm2.meanSquaredError)
println("MAE: " + rm2.meanAbsoluteError)
println("RMSE Squared: " + rm2.rootMeanSquaredError)
println("R Squared: " + rm2.r2)
println("Explained Variance: " + rm2.explainedVariance + "\n")

// COMMAND ----------

display(holdout2.withColumnRenamed("attendance", "actual"))


// COMMAND ----------

import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}

import org.apache.spark.ml.evaluation.RegressionEvaluator

import org.apache.spark.ml.{Pipeline, PipelineStage}

val rfModel = new RandomForestRegressor()
  .setLabelCol("attendance")
  .setFeaturesCol("features")

val paramGrid = new ParamGridBuilder()
  .addGrid(rfModel.maxDepth, Array.range(1, 20))
  .addGrid(rfModel.numTrees, Array.range(1, 20))
  .build()

val steps:Array[PipelineStage] = Array(rfModel)

val pipeline = new Pipeline().setStages(steps)

val cv = new CrossValidator() // you can feel free to change the number of folds used in cross validation as well
  .setEstimator(pipeline) // the estimator can also just be an individual model rather than a pipeline
  .setEstimatorParamMaps(paramGrid)
  .setEvaluator(new RegressionEvaluator().setLabelCol("attendance"))

val pipelineFitted = cv.fit(training2)

// COMMAND ----------

println("The Best Parameters:\n--------------------")
println(pipelineFitted.bestModel.asInstanceOf[org.apache.spark.ml.PipelineModel].stages(0))
pipelineFitted
  .bestModel.asInstanceOf[org.apache.spark.ml.PipelineModel]
  .stages(0)
  .extractParamMap

// COMMAND ----------

val holdout3 = pipelineFitted.bestModel
  .transform(test2)
  .selectExpr("home","visitor","prediction", "attendance")
display(holdout3)

// COMMAND ----------

//Look at regression metrics

val rm3 = new RegressionMetrics(
  holdout3.select("prediction", "attendance").rdd.map(x =>
  (x(0).asInstanceOf[Double], x(1).asInstanceOf[Int])))

println("MSE: " + rm3.meanSquaredError)
println("MAE: " + rm3.meanAbsoluteError)
println("RMSE Squared: " + rm3.rootMeanSquaredError)
println("R Squared: " + rm3.r2)
println("Explained Variance: " + rm3.explainedVariance + "\n")

// COMMAND ----------

display(holdout3.withColumnRenamed("attendance", "actual"))


// COMMAND ----------

//Count number of times team play eachother. Range is about 5 - 20
nhl_no_outliers.groupBy("home", "visitor").count().show(1000, false)

// COMMAND ----------

//Attendance by date including outlier
val maple_leafs = nhl.filter("home == 'Toronto Maple Leafs'")
display(maple_leafs)

// COMMAND ----------

val bruins = nhl.filter("home == 'Boston Bruins'")
display(bruins)

// COMMAND ----------

val kings = nhl.filter("home == 'Los Angeles Kings'")
display(kings)

// COMMAND ----------

//Investigation of rivarly impact on attendance. The Senators average attendance has declines over the past five years. However, the attendance is much higher when the Leafs are in town. 
// Teams from http://www.rivalrywatch.com/Home/Team/420 and https://www.hockeyfights.com/teams/21/team_rivals
display(nhl_no_outliers.filter($"home" === "Ottawa Senators").filter($"visitor" === "Toronto Maple Leafs").agg(avg("attendance")))

// COMMAND ----------

display(nhl_no_outliers.filter($"home" === "Ottawa Senators").filter($"visitor" === "Montreal Canadiens").agg(avg("attendance")))

// COMMAND ----------

display(nhl_no_outliers.filter($"home" === "Ottawa Senators").filter($"visitor" === "Buffalo Sabres").agg(avg("attendance")))

// COMMAND ----------

display(nhl_no_outliers.filter($"home" === "Ottawa Senators").filter($"visitor" === "Boston Bruins").agg(avg("attendance")))

// COMMAND ----------

display(nhl_no_outliers.filter($"home" === "Ottawa Senators").filter($"visitor" !=="Toronto Maple Leafs").filter($"visitor" !=="Montreal Canadiens").filter($"visitor" !=="Buffalo Sabres").filter($"visitor" !=="Boston Bruins").agg(avg("attendance")))
