package se.kth.spark.lab1.task5

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
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.feature.PolynomialExpansion

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
    
    //create the polynomial expansion transformer
    val polynomialExpansionT = new PolynomialExpansion()
      .setInputCol("features").setOutputCol("polynomial_exp").setDegree(2)  
      
    //Split in training and test 
    val Array(training, test) = obsDF.randomSplit(Array(0.9, 0.1), seed = 12345)

    //set up as input feature the 2-way interactions
    val myLR = new LinearRegression().setFeaturesCol("polynomial_exp")
    
    //create pipeline and set stages
    val pipeline = new Pipeline().setStages(Array(regexTokenizer,arr2Vect,lSlicer,v2d,lShifter,fSlicer,polynomialExpansionT,myLR))
    val lrStage = 7
      
    //Use a ParamGridBuilder to construct a grid of parameters to search over.
    val paramGrid = new ParamGridBuilder()
      .addGrid(myLR.regParam, Array(0.005, 0.05, 0.08, 0.1, 0.15, 0.3, 0.5))
      .addGrid(myLR.maxIter, Array(3, 5, 8, 10, 25, 50, 75))
      .build()
      
    //Setup CrossValidator, CrossValidator tries all combinations of values and determine best model using the evaluator.
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new RegressionEvaluator)
      .setEstimatorParamMaps(paramGrid)
    
    print("Training models...")
    val cvModel: CrossValidatorModel = cv.fit(training)
    val lrModel = cvModel.bestModel.asInstanceOf[PipelineModel].stages(lrStage).asInstanceOf[LinearRegressionModel]
    
    //Make predictions from the best model
    val predictions = cvModel.bestModel.transform(test)
      .select("prediction", "label", "polynomial_exp")
    //print first 10
    predictions.show(10)  
    
    //use evaluator to compute rmse on test set
    val evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
      
    val test_rmse = evaluator.evaluate(predictions)
    val train_rmse = lrModel.summary.rootMeanSquaredError
    
    //Additional : print the learned linear regression model
    println("Best learned linear regression model that captures the 2-way interactions:\n" +
	    "regParam = " + lrModel.getRegParam + '\n' +
	    "maxIter = " + lrModel.getMaxIter)    
    
    //print best model RMSE to compare to previous
    println("RMSE on test data = " + test_rmse)
    println("RMSE on train data = " + train_rmse)
    
  }
}