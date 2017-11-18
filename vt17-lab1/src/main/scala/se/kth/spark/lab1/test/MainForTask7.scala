package se.kth.spark.lab1.test

import se.kth.spark.lab1._
import org.apache.spark._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorSlicer
import org.apache.spark.sql.{SQLContext}
import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.functions.array
import scala.collection.Seq

//Local version for HugeDataset LR
object MainForTask7 {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._
    
    val filePath = "src/main/resources/SongCSV.csv"
    
    var df = sqlContext.read.format("CSV").option("header","true").option("mode","DROPMALFORMED").load(filePath)
    df.show(5)
    
    val numericalFeatures: Seq[String] = Seq("Year","SongNumber","SongID","AlbumID","AlbumName","ArtistID","ArtistLatitude",
        "ArtistLocation","ArtistLongitude","ArtistName","Danceability","Duration","KeySignature","KeySignatureConfidence",
        "Tempo","TimeSignature","TimeSignatureConfidence","Title")
    var dfNum = df.select(numericalFeatures.map(name => col(name)):_*)
    
    //removing columns with missing values, categorical features and filtering where year is > 0
    dfNum = dfNum.drop("SongNumber","SongID","AlbumID","AlbumName","ArtistID","ArtistLatitude",
        "ArtistLocation","ArtistLongitude","ArtistName","Danceability","Title")
    dfNum = dfNum.select(array($"Year", $"Duration", $"KeySignature", $"KeySignatureConfidence",
        $"Tempo", $"TimeSignature", $"TimeSignatureConfidence")as "values").filter($"Year" >0)
    dfNum.show(5)
    //-------------------------------------------------------------------------------------
    /* TRANSFORMERS TASK 2*/
    
    /**** No need because is already an array
    val regexTokenizer = new RegexTokenizer()
      .setInputCol("values") 
      .setOutputCol("array_features") 
      .setPattern("\\,") 
    **************************************/     
    val arr2Vect = new Array2Vector()
      .setInputCol("values")  
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
      .setIndices(Array(1,2,3,4,5,6))
      
    //------------------------------------------------------------------------------------- 

    //split dataset in training and test
    val Array(training, test) = dfNum.randomSplit(Array(0.7, 0.3), seed=1515)  
        
    //using best parameter estimated in Task4 with Cross validation
    val myLR = new LinearRegression()
      .setMaxIter(3)
      .setRegParam(0.1)
      .setElasticNetParam(0.1)
     
    val pipeline = new Pipeline().setStages(Array(arr2Vect,lSlicer,v2d,lShifter,fSlicer,myLR))
    val lrStage = 5 //index of myLR 
    val pipelineModel: PipelineModel = pipeline.fit(training)
    val lrModel = pipelineModel.stages(lrStage).asInstanceOf[LinearRegressionModel]

    //print rmse of our model
    val trainingSummary = lrModel.summary
    println(s"RMSE on training: ${trainingSummary.rootMeanSquaredError}")
    
    //do prediction - print first k
    val predictions = pipelineModel.transform(test)
    predictions.select("prediction", "label", "features")
    
    //use evaluator to compute rmse on test set
    val evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
      
    val test_rmse = evaluator.evaluate(predictions)
     
    println("RMSE on test data = " + test_rmse)

    
  }
}