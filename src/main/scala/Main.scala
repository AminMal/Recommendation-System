import dataframes.DataframeLoader
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.log4j.{Level, Logger}

object Main extends App {

  Logger.getLogger("org").setLevel(Level.ERROR)
  System.setProperty("hadoop.home.dir", "C:\\Users\\lenovo\\hadoop")
  val spark = SparkSession.builder()
    .master("local[4]")
    .appName("recommenderV2")
    .getOrCreate()
// D:/Daneshgah/final project/code/recommenderV2/src/main/resources/data/data/users.dat
  val loader = DataframeLoader(spark)
//  loader.getRatings().show()
//  loader.getMovies().show()
//  loader.getUsers().show()

  val Array(trainingData, testData) = loader.getRatings().randomSplit(Array(0.6, 0.4))
  val layers = Array(11, 5, 4, 4, 3 , 7)

  val trainer = new MultilayerPerceptronClassifier()
    .setLayers(layers)
    .setBlockSize(128)
    .setSeed(1234L)
    .setMaxIter(100)

  // new df: rating2. (user_id, movie_id, timestamp, rating, user_age, user_language, movie_director, movie_DOR)
  //                       8  ,  10
  // Ahmad, Gholami, 25, farsi
  // 1999, XDirector,
  val users = loader.getUsers()
  val movies = loader.getMovies()
  val ratings = loader.getRatings()

  val ratings2 = ratings
    .join(users, ratings.col("UserID") === users.col("UserID"), "inner")
    .join(movies, ratings.col("MovieID") === movies.col("MovieID"), "inner")
    .select(
      ratings.col("UserID") as "UserID",
      ratings.col("MovieID") as "MovieID",
      ratings.col("Rating") as "Rating",
      users.col("Gender") as "Gender",
      users.col("Age") as "Age",
      movies.col("Genres") as "Genres"
    )

  ratings2.show()

//  ratings2.write
//      .mode(SaveMode.ErrorIfExists)
//      .csv("src\\main\\resources\\data\\rates_2")

//  val model = trainer.fit(trainingData)
//  val result = model.transform(testData)
//  val predictionAndLabels = result.select("prediction", "label")
//  val evaluator = new MulticlassClassificationEvaluator()
//    .setMetricName("accuracy")
//
//  println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")
  System.in.read()

}
