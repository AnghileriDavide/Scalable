package se.kth.spark.lab1.task3

import org.apache.spark._
import org.apache.spark.sql.{ SQLContext, DataFrame }
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.feature.VectorAssembler

import se.kth.spark.lab1._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorSlicer
import org.apache.spark.sql.functions.split
import org.apache.spark.sql.functions._
import java.util.function.ToDoubleFunction
import org.apache.spark.ml.feature.VectorSlicer

object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._

    //Load and convert Dataset as in Task2
    val filePath = "src/main/resources/millionsong.txt"
    val rawDF = sqlContext.read.text(filePath)

    val regexTokenizer = new RegexTokenizer()
      .setInputCol("value")
      .setOutputCol("features")
      .setPattern("\\,")
    val tokenized = regexTokenizer.transform(rawDF).select("features")

    val arr2Vect = new Array2Vector()
      .setInputCol("features")
      .setOutputCol("feature_vector")
    val vectorized = arr2Vect.transform(tokenized).select("feature_vector")

    val lSlicer = new VectorSlicer()
      .setInputCol("feature_vector")
      .setOutputCol("year_vector")
      .setIndices(Array(0))
    val labeled = lSlicer.transform(vectorized)

    val v2d = new Vector2DoubleUDF(v => v(0))
      .setInputCol("year_vector")
      .setOutputCol("year_double")
    val double_labeled = v2d.transform(labeled)

    val min_year: Double = double_labeled.agg(min("year_double")).first().toSeq.asInstanceOf[Seq[Double]](0)
    val lShifter = new DoubleUDF(x => x - min_year)
      .setInputCol("year_double")
      .setOutputCol("year_shifted")

    val fSlicer = new VectorSlicer()
      .setInputCol("feature_vector")
      .setOutputCol("selected_features")
      .setIndices(Array(1, 2, 3))

    //Starting task 3 - Linear Regression - input: f1,f2,f3 - output: year
    val myLR = new LinearRegression()
      .setMaxIter(10)
      .setRegParam(0.1)
      .setElasticNetParam(0.1)
      .setFeaturesCol("selected_features")
      .setLabelCol("year_shifted")

    val Array(training, test) = rawDF.randomSplit(Array(0.7, 0.3))

    val lrStage = 6
    val pipeline = new Pipeline().setStages(Array(regexTokenizer, arr2Vect, lSlicer, v2d, lShifter, fSlicer, myLR))
    val pipelineModel: PipelineModel = pipeline.fit(training)
    val lrModel = pipelineModel.stages(lrStage).asInstanceOf[LinearRegressionModel]

    //print rmse of our model
    val ModelSummary = lrModel.summary
    println("RMSE: " + ModelSummary.rootMeanSquaredError)

    //do prediction - print first k
    val predictions = pipelineModel.transform(test)
    val k = 20
    predictions.select("prediction", "year_shifted").show(k)

  }
}
