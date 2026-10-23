name := "api-spark-scala"

version := "1.0"

//Version de Scala
scalaVersion := "2.13.17"

//Version de Spark
val sparkVersion = "4.1.1" 

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion % "provided",
  "org.apache.spark" %% "spark-sql"  % sparkVersion % "provided",
  "org.apache.spark" %% "spark-sql-kafka-0-10" % sparkVersion
)
