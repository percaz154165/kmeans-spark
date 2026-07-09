package readers;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.util.ArrayList;
import java.util.List;

import static org.apache.spark.sql.functions.*;

public class ReadDatasetSpark {

    public static Dataset<Row> loadDataset(SparkSession spark, String path) {

        System.out.println("Loading dataset from: " + path);

        Dataset<Row> dataset = spark.read()
                .option("header", true)
                .option("inferSchema", true)
                .csv(path);

        System.out.println("Dataset loaded.");
        System.out.println("Rows: " + dataset.count());

        String[] columns = dataset.columns();

        List<org.apache.spark.sql.Column> minMaxExpr = new ArrayList<>();

        for (String c : columns) {
            minMaxExpr.add(min(c).alias(c + "_min"));
            minMaxExpr.add(max(c).alias(c + "_max"));
        }

        Row stats = dataset.agg(
                minMaxExpr.get(0),
                minMaxExpr.subList(1, minMaxExpr.size()).toArray(new org.apache.spark.sql.Column[0])
        ).first();

        Dataset<Row> normalized = dataset;

        for (int i = 0; i < columns.length; i++) {

            String col = columns[i];

            double minValue = ((Number) stats.get(2 * i)).doubleValue();
            double maxValue = ((Number) stats.get(2 * i + 1)).doubleValue();

            normalized = normalized.withColumn(
                    col,
                    col(col).minus(minValue)
                            .divide(maxValue - minValue)
            );
        }

        System.out.println("Dataset normalized.");

        return normalized;
    }
}