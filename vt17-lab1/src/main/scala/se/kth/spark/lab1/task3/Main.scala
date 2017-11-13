package se.kth.spark.lab1.task3

import se.kth.spark.lab1._

import org.apache.spark._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.feature.VectorSlicer
import org.apache.spark.sql.{SQLContext, DataFrame}
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.regression.LinearRegression

object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._
    
    val filePath = "src/main/resources/millionsong.txt"
    val obsDF: DataFrame = sqlContext.read.text(filePath)
    
    //-------------------------------------------------------------------------------------
    /* TRANSFORMERS TASK 2*/
    
    val regexTokenizer = new RegexTokenizer()
      .setInputCol("value") 
      .setOutputCol("array_features") 
      .setPattern("\\,") 
        
    val arr2Vect = new Array2Vector()
      .setInputCol("array_features")  
      .setOutputCol("vector_features")
      
    val lSlicer = new VectorSlicer()  
      .setInputCol("vector_features")
      .setOutputCol("year_vector")
      .setIndices(Array(0))
        
    val v2d = new Vector2DoubleUDF(v => v(0))
      .setInputCol("year_vector")
      .setOutputCol("year_double")
    
    val min_year : Double = 1922
      
    val lShifter = new DoubleUDF(x => x - min_year)
      .setInputCol("year_double")
      .setOutputCol("label")
      
    val fSlicer = new VectorSlicer()
      .setInputCol("vector_features")
      .setOutputCol("features")
      .setIndices(Array(1,2,3))
      
    //------------------------------------------------------------------------------------- 

    //split dataset in training and test
    val Array(training, test) = obsDF.randomSplit(Array(0.7, 0.3))  
        
    val myLR = new LinearRegression()
      .setMaxIter(10)
      .setRegParam(0.1)
      .setElasticNetParam(0.1)
    
    
    val pipeline = new Pipeline().setStages(Array(regexTokenizer,arr2Vect,lSlicer,v2d,lShifter,fSlicer,myLR))
    val lrStage = 6 //index of myLR 
    val pipelineModel: PipelineModel = pipeline.fit(training)
    val lrModel = pipelineModel.stages(lrStage).asInstanceOf[LinearRegressionModel]

    //print rmse of our model
    val trainingSummary = lrModel.summary
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    
    //do prediction - print first k
    val predictions = pipelineModel.transform(test)
    predictions.select("prediction", "label", "features").show(10)
    
  }
}