package se.kth.spark.lab1.task1

import se.kth.spark.lab1._

import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.Pipeline
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql._
import org.apache.spark.sql.types._

object Main {
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
    rawDF.limit(5).collect().foreach(println)
    //The delimiter is: ","
    println("Number of features: " + rawDF.columns.length)
    //Number of features: 13
    println("Data types: " + rawDF.first().schema)
    //The data types are all Strings

    //Step2: split each row into an array of features
    val recordsRdd = rdd.map(x => x.split("\\,"))

    //Step3: map each row into a Song object by using the year label and the first three features
    val songsRdd = recordsRdd.map(x => (x(0).toFloat, x(1).toFloat, x(2).toFloat, x(3).toFloat) )

    //Step4: convert your rdd into a datafram
    val songsDfTmp = songsRdd.toDF("year", "f1", "f2", "f3")
    // cast year into Int
    val songsDf =songsDfTmp.withColumn("yearTmp", 'year.cast("Int"))
    .drop("year")
    .withColumnRenamed("yearTmp", "year")

    //QUESTIONS:
    //1. How many songs there are in the DataFrame?
    println("There are " + songsDf.count() + " songs")
    //2. How many songs were released between the years 1998 and 2000?
    println( songsDf.filter($"year" > 1997 and $"year" < 2001 ).count() + " songs were released between the years 1998 and 2000")
    //3. What is the min, max and mean value of the year column?
    songsDf.describe("year").show()
    //4. Show the number of songs per year between the years 2000 and 2010?
    songsDf.filter($"year" > 1999 and $"year" < 2011 ).groupBy("year").count().show()
  }
}