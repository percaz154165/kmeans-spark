package main;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vector;

import readers.ReadDatasetSpark;

public class MainSparkKMeans {

    public static void main(String[] args) {

        SparkSession spark = SparkSession.builder()
                .appName("KMeansSparkML")
                .master("local[*]")
                .getOrCreate();

        spark.sparkContext().setLogLevel("ERROR");

        long t0 = System.nanoTime();

        Dataset<Row> dataset = ReadDatasetSpark.loadDataset(
                spark,
                "src/main/java/DATA/datos_ampliado.csv"
        );
        long t1 = System.nanoTime();

        String[] inputColumns = {"age", "membership_years", "number_of_children", "quantity", "unit_price", "discount_applied", "avg_purchase_value", "avg_discount_used", "online_purchases", "in_store_purchases", "avg_items_per_transaction", "avg_transaction_value", "total_returned_items", "total_returned_value", "total_sales", "total_transactions", "total_items_purchased", "total_discounts_received", "avg_spent_per_category", "max_single_purchase_value", "min_single_purchase_value", "product_review_count", "product_return_rate", "product_weight", "distance_to_store", "customer_support_calls", "days_since_last_purchase"}; 
        
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(inputColumns)
                .setOutputCol("features"); 
        Dataset<Row> assembledData = assembler.transform(dataset);

        KMeans kmeans = new KMeans()
                .setK(6)
                .setMaxIter(100)
                .setSeed(42);

        KMeansModel model = kmeans.fit(assembledData);

        long t2 = System.nanoTime();


        System.out.println("Rows: " + dataset.count());
        System.out.println("Tiempo de ejecución (segundos): " + (t2 - t0) / 1e9);
        System.out.println("Tiempo de lectura del dataset (segundos): " + (t1 - t0) / 1e9);
        System.out.println("Tiempo de entrenamiento del modelo (segundos): " + (t2 - t1) / 1e9);
        Vector[] centers = model.clusterCenters();
        System.out.println("\n--- Centroides de los Clusters ---");
        for (int i = 0; i < centers.length; i++) {
            System.out.println("Cluster " + i + ": " + centers[i]);
        }

        Dataset<Row> predictions = model.transform(assembledData);
        
        System.out.println("\n--- Muestra de los datos con sus clusters asignados ---");
        predictions.select("features", "prediction").show(10);

        spark.stop();
    }
}