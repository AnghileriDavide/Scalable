package se.kth.spark.lab1.task2

import se.kth.spark.lab1._

import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorSlicer
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions.split
import org.apache.spark.sql.functions._
import java.util.function.ToDoubleFunction


object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._

    val filePath = "src/main/resources/millionsong.txt"
    val rawDF = sqlContext.read.text(filePath)  
    
    println("raw DF")
    rawDF.show(5)
    
    //Step1: tokenize each row
    val regexTokenizer = new RegexTokenizer()
      .setInputCol("value") 
      .setOutputCol("features") 
      .setPattern("\\,") 
     
    //Step2: transform with tokenizer and show 5 rows
    val tokenized = regexTokenizer.transform(rawDF).select("features") 
		tokenized.show(5)
    
    //Step3: transform array of tokens to a vector of tokens (use our ArrayToVector)
    val arr2Vect = new Array2Vector()
      .setInputCol("features") 
      .setOutputCol("feature_vector")
    val vectorized = arr2Vect.transform(tokenized).select("feature_vector")
    	
    //Step4: extract the label(year) into a new column
    val lSlicer = new VectorSlicer()  
      .setInputCol("feature_vector")
      .setOutputCol("year_vector")
      .setIndices(Array(0))
    
    val labeled = lSlicer.transform(vectorized)  
    labeled.show()
   
    //Step5: convert type of the label from vector to double (use our Vector2Double)
    val v2d = new Vector2DoubleUDF(v => v(0))
      .setInputCol("year_vector")
      .setOutputCol("year_double")
    
    val double_labeled = v2d.transform(labeled)
    double_labeled.show()
    
    //Step6: shift all labels by the value of minimum label such that the value of the smallest becomes 0 (use our DoubleUDF)  
    val min_year : Double = double_labeled.agg(min("year_double")).first().toSeq.asInstanceOf[Seq[Double]](0)
    println(min_year)
   
    val lShifter = new DoubleUDF(x => x - min_year )
      .setInputCol("year_double")
      .setOutputCol("year_shifted")
    
    val shifted = lShifter.transform(double_labeled)
    shifted.show()
   
    //Step7: extract just the 3 first features in a new vector column
    val fSlicer = new VectorSlicer()
      .setInputCol("feature_vector")
      .setOutputCol("selected_features")
      .setIndices(Array(1,2,3))
      
    val sliced = fSlicer.transform(shifted)  
    sliced.show()
      
    //Step8: put everything together in a pipeline
    val pipeline = new Pipeline().setStages(Array(regexTokenizer,arr2Vect,lSlicer,v2d,lShifter,fSlicer))
    
    //Step9: generate model by fitting the rawDf into the pipeline
    val pipelineModel = pipeline.fit(rawDF)
    
    //Step10: transform data with the model - do predictions
    pipelineModel.transform(rawDF)
    //Step11: drop all columns from the dataframe other than label and features
      .select("selected_features","year_shifted")
      .show(10)  
  }
}