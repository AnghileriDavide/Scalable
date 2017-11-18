package se.kth.spark.lab1.task6

import se.kth.spark.lab1._

import org.apache.spark._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{ Row, SQLContext, DataFrame }
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.feature.VectorSlicer
import org.apache.spark.ml.evaluation.RegressionEvaluator

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
    val Array(training, test) = obsDF.randomSplit(Array(0.9, 0.1))  
     

    val myLR = new MyLinearRegressionImpl()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setPredictionCol("prediction")
      
    val pipeline = new Pipeline().setStages(Array(regexTokenizer,arr2Vect,lSlicer,v2d,lShifter,fSlicer,myLR))
    val lrStage = 6
    val pipelineModel: PipelineModel = pipeline.fit(training)
    val myLRModel = pipelineModel.stages(lrStage).asInstanceOf[MyLinearModelImpl]

    
    //Make predictions and print first 10
    val predictions = pipelineModel.transform(test)
      .select("prediction", "label", "features")
      
    predictions.show(10)  
    
    //use evaluator to compute rmse on test set
    val evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
      
    val test_rmse = evaluator.evaluate(predictions)
    //last elem of the trainingError array
    //if you want to look at all the "history" of the error print the entire array
    val train_rmse = myLRModel.trainingError.last
     
    //print best model RMSE to compare to previous
    println("RMSE on test data = " + test_rmse)
    println("RMSE on train data = " + train_rmse)

   
  }
}