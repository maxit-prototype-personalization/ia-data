import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.streaming.Trigger

object MainStreamingPersonalization {
  def main(args: Array[String]): Unit = {

    // Initialisation de la Session Spark avec support HDFS
    val spark = SparkSession.builder()
      .appName("Spark Streaming Kafka Personalisation")
      .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000")
      .config("spark.driver.host", "127.0.0.1")
      .config("spark.driver.bindAddress", "127.0.0.1")
      .config("spark.ui.enabled", "false")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._
    spark.sparkContext.setLogLevel("WARN")

    // DÉMARRAGE DU STREAMING SPARK KAFKA -> HDFS
    println("==================================================")
    println("      DÉMARRAGE SPARK STREAMING (KAFKA -> HDFS)   ")
    println("==================================================")

    // =========================================================================
    // LECTURE DU REFERENTIEL MAPPING DEPUIS HDFS
    // =========================================================================
    val mapPath = "hdfs://localhost:9000/mapping/usages_maxit.csv"
    var dfMapping = spark.read.option("header", "true").option("inferSchema", "true").csv(mapPath)

    // Normalisation des colonnes du mapping (remplacement des espaces par "_" et passage en minuscule)
    for (colName <- dfMapping.columns) {
      val newName = colName.replace(" ", "_").toLowerCase()
      dfMapping = dfMapping.withColumnRenamed(colName, newName)
    }

    dfMapping = dfMapping
      .withColumn("id_services", trim(col("id_services")))
      .withColumn("univers", upper(trim(col("univers"))))
      .withColumn("mode", upper(trim(col("mode"))))
      // === 2. FILTRAGE DES DONNÉES INVALIDES (Mapping) ===
      .filter(col("id_services").isNotNull && col("id_services") =!= "")
      // === 4. SUPPRESSION DES DOUBLONS (Mapping) ===
      .dropDuplicates("id_services")

    // =========================================================================
    // 1. LECTURE DU FLUX TEMPS REEL DEPUIS APACHE KAFKA
    // =========================================================================
    val kafkaStream = spark.readStream
      .format("kafka")
      .option("kafka.bootstrap.servers", "localhost:9092")
      .option("subscribe", "tracking-events")
      .option("startingOffsets", "latest")
      .option("failOnDataLoss", "false")
      .load()

    // === 2. LECTURE DE LA STRUCTURE DES ÉVÉNEMENTS (Schéma JSON) ===
    // Définition du schéma décrivant la structure exacte des événements publiés par tracking-service
    val trackingEventSchema = new StructType()
      .add("id", StringType)
      .add("eventType", StringType)
      .add("msisdn", StringType)
      .add("userId", StringType)
      .add("userRole", StringType)
      .add("sourceService", StringType)
      .add("timestamp", StringType)
      .add("payload", MapType(StringType, StringType))

    // Deserialisation du message Kafka JSON en colonnes structurées
    val dfEvents = kafkaStream
      .selectExpr("CAST(value AS STRING) as json_value")
      .select(from_json(col("json_value"), trackingEventSchema).as("data"))
      .select("data.*")
      // === 3. FILTRAGE DES DONNÉES INVALIDES (Événements sans MSISDN) ===
      .filter(col("msisdn").isNotNull && col("msisdn") =!= "")

    // =========================================================================
    // 4. AGRÉGATION DES COMPORTEMENTS UTILISATEURS
    // =========================================================================
    // Extraction du mode de paiement, du mode IHM et de l'univers IHM depuis le payload
    val dfEventsWithPayment = dfEvents
      .withColumn("payment_method", col("payload").getItem("paymentMethod"))
      .withColumn("ui_mode", col("payload").getItem("mode"))
      .withColumn("ui_universe", col("payload").getItem("universe"))

    // Mapping des types d'événements Kafka vers les service_id du mapping CSV (OMY vs TELCO)
    val dfEventsMapped = dfEventsWithPayment.withColumn("service_id", when(col("eventType") === "DEPOT", "DEPOT")
      .when(col("eventType") === "RETRAIT", "OMY.SERVICES.RETRAIT")
      .when(col("eventType") === "TRANSFERT", "OMY.SERVICES.TRANSFERT")
      .when(col("eventType") === "ACHAT_RAPIDO", "RAPIDO")
      // Si l'utilisateur effectue l'achat depuis l'univers OMY (IHM), on le mappe sur OMY.SERVICES.VOICEBUNDLE
      .when(col("ui_universe") === "OMY" && (col("eventType").startsWith("ACHAT_PASS_") || col("eventType") === "ACHAT_CREDIT"), "OMY.SERVICES.VOICEBUNDLE")
      // Sinon (univers TELCO ou par défaut)
      .when(col("eventType") === "ACHAT_PASS_ILLIFLEX", "TELCO.SERVICES.PASS.ILLIFLEX")
      .when(col("eventType") === "ACHAT_PASS_ILLIMIX", "TELCO.SERVICES.PASS.ILLIMIX")
      .when(col("eventType") === "ACHAT_PASS_INTERNET", "TELCO.SERVICES.PASS.DATA")
      .when(col("eventType") === "ACHAT_CREDIT", "TELCO.SERVICES.PASS.VOICE")
      .when(col("eventType") === "ACHAT_INTERNATIONAL", "TELCO.SERVICES.PASS.INTERNATIONAL")
      
      // Nouveaux services OMY
      .when(col("eventType") === "ACHAT_MASTERCARD" || col("eventType") === "OMY.SERVICES.MASTERCARD", "OMY.SERVICES.MASTERCARD")
      .when(col("eventType") === "ACCESS_FINANCIERS" || col("eventType") === "OMY.SERVICES.FINANCIERS", "OMY.SERVICES.FINANCIERS")
      .when(col("eventType") === "ACHAT_PI" || col("eventType") === "OMY.SERVICES.PI", "OMY.SERVICES.PI")
      .when(col("eventType") === "TRANSFERT_INTER_COMPTE" || col("eventType") === "OMY.SERVICES.INTER_COMPTE", "OMY.SERVICES.INTER_COMPTE")
      .when(col("eventType") === "PAIEMENT_FACTURE" || col("eventType") === "OMY.SERVICES.BILLERS", "OMY.SERVICES.BILLERS")
      .when(col("eventType") === "ACHAT_TER" || col("eventType") === "OMY.SERVICES.BILLERS.TER", "OMY.SERVICES.BILLERS.TER")
      .when(col("eventType") === "ACCES_TIKTAK" || col("eventType") === "OBA.SERVICES.TIKTAK", "OBA.SERVICES.TIKTAK")
      .when(col("eventType") === "ACHAT_CANALPLUS", "CANALPLUS")
      .when(col("eventType") === "ACHAT_FLEXEAU", "FLEXEAU")
      .when(col("eventType") === "ACHAT_SENELEC" || col("eventType") === "SENELEC_MAIN_PAGE", "SENELEC_MAIN_PAGE")

      // Nouveaux services TELCO
      .when(col("eventType") === "ACHAT_HOME_INTERNET" || col("eventType") === "TELCO.SERVICES.HOME_INTERNET", "TELCO.SERVICES.HOME_INTERNET")
      .when(col("eventType") === "ACCES_LOYALTY" || col("eventType") === "TELCO.SERVICES.LOYALTY", "TELCO.SERVICES.LOYALTY")
      .when(col("eventType") === "ACCES_SOS_CREDIT" || col("eventType") === "TELCO.SERVICES.SOS_CREDIT", "TELCO.SERVICES.SOS_CREDIT")
      .when(col("eventType") === "ACCES_P2P_BONUS" || col("eventType") === "TELCO.SERVICES.P2P_BONUS", "TELCO.SERVICES.P2P_BONUS")
      .when(col("eventType") === "ACHAT_PASS_MIXEL" || col("eventType") === "TELCO.SERVICES.PASS.MIXEL", "TELCO.SERVICES.PASS.MIXEL")
      .when(col("eventType") === "ACCES_SONATEL" || col("eventType") === "SONATELF_MAIN_PAGE", "SONATELF_MAIN_PAGE")
      .when(col("eventType") === "ACHAT_PASS_TRAVEL" || col("eventType") === "TELCO.SERVICES.PASS.TRAVEL", "TELCO.SERVICES.PASS.TRAVEL")
      .when(col("eventType") === "P2P_TRANSFER" || col("eventType") === "TELCO.SERVICES.P2P", "TELCO.SERVICES.P2P")
      .when(col("eventType") === "ACHAT_PASS_WIDO" || col("eventType") === "TELCO.SERVICES.PASS.WIDO", "TELCO.SERVICES.PASS.WIDO")
      .when(col("eventType") === "ACHAT_DALAL" || col("eventType") === "TELCO.SERVICES.LEISURE.DALAL", "TELCO.SERVICES.LEISURE.DALAL")
      .when(col("eventType") === "ACCES_SOS_PASS" || col("eventType") === "TELCO.SERVICES.SOS_PASS", "TELCO.SERVICES.SOS_PASS")

      .otherwise(col("eventType"))
    )

    // Jointure Stream-Statique avec le DataFrame de mapping
    val dfEnrichi = dfEventsMapped.join(
      dfMapping, dfEventsMapped("service_id") === dfMapping("id_services"), "left"
    )

    // =========================================================================
    // 5. SUPPRESSION DES DOUBLONS & 6. GÉNÉRATION DES PROFILS UTILISATEURS 
    // =========================================================================
    val parsedTimestamp = coalesce(
      to_timestamp(from_unixtime(col("timestamp").cast(DoubleType))),
      to_timestamp(col("timestamp")),
      current_timestamp()
    )

    val dfFinal = dfEnrichi.select(
      col("msisdn"),
      coalesce(col("ui_universe"), dfMapping("univers"), lit("OMY")).as("univers"),
      coalesce(col("ui_mode"), dfMapping("mode"), lit("SIMPLE")).as("mode"),
      // === 5. SUPPRESSION DES DOUBLONS DANS LA LISTE DE SERVICES ===
      array_distinct(array(col("service_id"))).as("liste_de_services"),
      date_format(parsedTimestamp, "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'").as("last_date"),
      date_format(current_timestamp(), "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'").as("date")
    )

    // =========================================================================
    // 7. ÉCRITURE DANS HDFS & 8. ACTUALISATION DES DONNÉES (Trigger Batch)
    // =========================================================================
    val cheminExportJson = "hdfs://localhost:9000/api_output_json"
    val checkpointPath = "hdfs://localhost:9000/checkpoint/personalisation"

    val query = dfFinal.writeStream
      .format("json")
      // === 7. ÉCRITURE DANS HDFS ===
      .option("path", cheminExportJson)
      .option("checkpointLocation", checkpointPath)
      .outputMode("append")
      // === 8. ACTUALISATION DES DONNÉES (Fréquence du micro-batch Spark Streaming) ===
      .trigger(Trigger.ProcessingTime("300 seconds"))
      .start()

    query.awaitTermination()
  }
}
