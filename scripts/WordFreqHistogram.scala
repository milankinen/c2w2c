import scala.io.Source

val words = Source.fromFile(args(0))
  .getLines()
  .flatMap(_.split(" "))
  .toSeq

words
  .groupBy(w => ((w.length - 1) / 5.0).toInt)
  .mapValues(_.size)
  .toSeq
  .sortBy(_._1)
  .foreach { case (l, n) =>
    val i = l * 5 + 1
    println(s"$i-${i + 4} : $n")
  }



