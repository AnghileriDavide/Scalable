package se.kth.spark.lab1.task1

import se.kth.spark.lab1._

import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.Pipeline
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions._

object Main {
  case class Song (year : Int, f1 : Double, f2 : Double, f3 : Double)
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._

    val filePath = "src/main/resources/millionsong.txt"
    val rawDF = sqlContext.read.csv(filePath)
 
    val rdd = sc.textFile(filePath)

    //Step1: print the first 5 rows, what is the delimiter, number of features and the data types?
    // delimiter: ","
    // number of features: 13
    // data types: double
    println(rdd.take(5).mkString("\n"))
    
    //Step2: split each row into an array of features
    val recordsRdd = rdd.map(line => line.split(","))
    
    //Step3: map each row into a Song object by using the year label and the first three features  
    val songsRdd = recordsRdd.map(array => Song(array(0).toDouble.toInt, array(1).toDouble,array(2).toDouble, array(3).toDouble) )

    //Step4: convert your rdd into a datafram
    val songsDf = songsRdd.toDF()

    songsDf.registerTempTable("Songs")
    
    //----------------------------------------------------------------------------------
    /* 1. How many songs there are in the DataFrame? */
    //----------------------------------------------------------------------------------
    
    //higher order function
    println(songsDf.count())
    //SQL query
    sqlContext.sql("select count(*) as Count from Songs").show()
    
    
    //----------------------------------------------------------------------------------
    /* 2. How many songs were released between the years 1998 and 2000? */
    //----------------------------------------------------------------------------------
    
    //higher order function
    println(songsDf.filter($"year".between(1998, 2000)).count())
    //SQL query
    sqlContext.sql("select count(*) as Count from Songs where year between 1998 and 2000").show()
    
    
    //----------------------------------------------------------------------------------
    /*3. What is the min, max and mean value of the year column? */
    //----------------------------------------------------------------------------------
    
    //higher order function
    songsDf.agg(min("year"),max("year"),mean("year")).show()
    //SQL query
    sqlContext.sql("select min(year) as Min, max(year) as Max, avg(year) as Avg from Songs").show()
    
    //----------------------------------------------------------------------------------
    /*4. Show the number of songs per year between the years 2000 and 2010? */
    //----------------------------------------------------------------------------------
    
    //higher order function
    songsDf.filter($"year".between(2000,2010)).groupBy("year").count().show()
    //SQL query
    sqlContext.sql("select year,count(*) as Count from Songs where year between 2000 and 2010 group by year").show()
    
  }
}