import dataframes.DataframeLoader
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.ml.classification.{LogisticRegression, MultilayerPerceptronClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.{VectorAssembler, VectorSizeHint}
import org.apache.spark.ml.linalg.{VectorUDT, Vectors}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.sql.types._


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

  // new df: rating2. (user_id, movie_id, timestamp, rating, user_age, user_language, movie_director, movie_DOR)
  //                       8  ,  10
  // Ahmad, Gholami, 25, farsi
  // 1999, XDirector,

  val users = loader.getUsers()
  val movies = loader.getMovies()
  val ratings = loader.getRatings()

  val genres: Set[String] = Set(
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
  )

  val getMovieGenresFeature = udf { (column: String) =>
    val movieGenres = column.split("\\s*\\|\\s*").toSet
    Vectors.dense {
      genres.map { genre =>
        if (movieGenres.contains(genre)) 1D else 0D
      }.toArray
    }
  }

  val separateByBar = udf { (column: String) =>
    Vectors.dense(
      column.split("\\s*\\|\\s*").map {
        case "Action" => 1
        case "Adventure" => 2
        case "Animation" => 3
        case "Children's" => 4
        case "Comedy" => 5
        case "Crime" => 6
        case "Documentary" => 7
        case "Drama" => 8
        case "Fantasy" => 9
        case "Film-Noir" => 10
        case "Horror" =>  11
        case "Musical" =>  12
        case "Mystery" =>  13
        case "Romance" =>  14
        case "Sci-Fi" =>   15
        case "Thriller" => 16
        case "War" =>      17
        case "Western" =>  18
      }.map(_.toDouble)
    )
  }

  val likesOrNot = udf { (rating: Double) =>
    rating > 0.5
  }

  val getGenderInt = udf { (column: String) =>
    column match {
      case value if value equalsIgnoreCase "m" => 0
      case _ => 1
    }
  }

  val ratings2 = ratings
    .join(users, ratings.col("UserID") === users.col("UserID"), "inner")
    .join(movies, ratings.col("MovieID") === movies.col("MovieID"), "inner")
    .select(
      ratings.col("UserID") cast IntegerType as "UserID",
      ratings.col("MovieID") cast IntegerType as "MovieID",
      ratings.col("Rating") cast DoubleType as "Rating",
      getGenderInt(users.col("Gender")) as "Gender",
      users.col("Occupation") cast "Int" as "Occupation" ,
      users.col("Age") cast "Int" as "Age",
      movies.col("Genres") as "Genres",
      likesOrNot(ratings.col("Rating")) cast IntegerType as "label",
      getMovieGenresFeature(ratings.col("Genres")) cast VectorType as "genres_feature"
    )

  ratings2.show()
  ratings2.printSchema()

//  ratings2.write
//      .mode(SaveMode.ErrorIfExists)
//      .csv("src\\main\\resources\\data\\rates_2")

  val assembler = new VectorAssembler()
    .setInputCols(Array("Gender","Age","Occupation", "genres_feature"))
    .setOutputCol("features")

//  val labelAssembler = new VectorAssembler()
//    .setInputCols(Array("UserID"))
//    .setOutputCol("label")

  val sizeHint = new VectorSizeHint()
    .setInputCol("features")
    .setHandleInvalid("skip")
    .setSize(4)

  val lr = new LogisticRegression()
    .setMaxIter(10)
    .setRegParam(0.3)
    .setElasticNetParam(0.8)
    .setFeaturesCol("features")   // setting features column
    .setLabelCol("label")       // setting label column

  //creating pipeline
  val pipeline = new Pipeline().setStages(Array(assembler,  lr))

  //fitting the model
  val Array(trainingData, testData) = ratings2.randomSplit(Array(0.4, 0.6))

  val lrModel = pipeline.fit(trainingData)

  val result = lrModel.transform(testData)

  val predictionAndLabels = result.select("Rating", "features", "label")
  val evaluator = new MulticlassClassificationEvaluator()
    .setPredictionCol("Rating")
    .setMetricName("accuracy")

  result.show(100, false)

  println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")

  //  val layers = Array(11, 5, 4, 4, 3 , 7)
//
//  val trainer = new MultilayerPerceptronClassifier()
//    .setLayers(layers)
//    .setBlockSize(128)
//    .setSeed(1234L)
//    .setMaxIter(100)

//  val model = trainer.fit(trainingData)
//  val result = model.transform(testData)
//  val predictionAndLabels = result.select("prediction", "label")
//  val evaluator = new MulticlassClassificationEvaluator()
//    .setMetricName("accuracy")
//
//  println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")
  System.in.read()

}
