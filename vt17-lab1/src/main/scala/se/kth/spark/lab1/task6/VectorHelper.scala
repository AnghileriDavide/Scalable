package se.kth.spark.lab1.task6

import org.apache.spark.ml.linalg.{Matrices, Vector, Vectors, DenseVector}

object VectorHelper {
  def dot(v1: Vector, v2: Vector): Double = {
     (0.0 /: (for ((a, b) <- v1.toArray zip v2.toArray) yield a * b)) (_ + _)
  }

  def dot(v: Vector, s: Double): Vector = {
    return new DenseVector(v.toArray.map(e => e * s))
  }

  def sum(v1: Vector, v2: Vector): Vector = {
    return new DenseVector(v1.toArray.zip(v2.toArray).map {case (a, b) => a + b})
  }

  def fill(size: Int, fillVal: Double): Vector = {
    return new DenseVector(Array.fill(size)(fillVal))
  }
  
}