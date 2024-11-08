from pyspark.sql import SparkSession
from pyspark.sql.functions import col, abs, unix_timestamp, stddev, row_number, when, count
from pyspark.sql.window import Window

# Create a Spark session
spark = SparkSession.builder.appName("Advanced Flight Data Analysis").getOrCreate()

# Load datasets
flights_df = spark.read.csv("flights.csv", header=True, inferSchema=True)
airports_df = spark.read.csv("airports.csv", header=True, inferSchema=True)
carriers_df = spark.read.csv("carriers.csv", header=True, inferSchema=True)

# Define output paths
output_dir = "output/"
task1_output = output_dir + "task1_largest_discrepancy.csv"
task2_output = output_dir + "task2_consistent_airlines.csv"
task3_output = output_dir + "task3_canceled_routes.csv"
task4_output = output_dir + "task4_carrier_performance_time_of_day.csv"

# ------------------------
# Task 1: Flights with the Largest Discrepancy Between Scheduled and Actual Travel Time
# ------------------------
def task1_largest_discrepancy(flights_df, carriers_df):
    # Calculate scheduled and actual travel times and the discrepancy
    flights_df = flights_df.withColumn("ScheduledTravelTime", 
                                       (unix_timestamp("ScheduledArrival") - unix_timestamp("ScheduledDeparture")) / 60)
    flights_df = flights_df.withColumn("ActualTravelTime", 
                                       (unix_timestamp("ActualArrival") - unix_timestamp("ActualDeparture")) / 60)
    flights_df = flights_df.withColumn("Discrepancy", 
                                       abs(col("ScheduledTravelTime") - col("ActualTravelTime")))

    # Join with carriers to get carrier names
    flights_df = flights_df.join(carriers_df, on="CarrierCode")

    # Window specification for ranking by discrepancy within each carrier
    window_spec = Window.partitionBy("CarrierCode").orderBy(col("Discrepancy").desc())

    # Add a row number based on the discrepancy ranking within each carrier
    flights_df = flights_df.withColumn("Rank", row_number().over(window_spec))

    # Select all rows with rankings to satisfy the requirement
    largest_discrepancy = flights_df.select(
        "FlightNum", "CarrierName", "Origin", "Destination", 
        "ScheduledTravelTime", "ActualTravelTime", "Discrepancy", "CarrierCode", "Rank"
    )

    # Write the result to a CSV file with overwrite mode
    largest_discrepancy.write.csv(task1_output, header=True, mode='overwrite')
    print(f"Task 1 output written to {task1_output}")

# ------------------------
# Task 2: Most Consistently On-Time Airlines Using Standard Deviation
# ------------------------
def task2_consistent_airlines(flights_df, carriers_df):
    flights_df.createOrReplaceTempView("flights")
    carriers_df.createOrReplaceTempView("carriers")
    
    query = """
        SELECT c.CarrierName, COUNT(f.FlightNum) AS NumberOfFlights,
               stddev(unix_timestamp(f.ActualDeparture) - unix_timestamp(f.ScheduledDeparture)) / 60 AS DepartureDelayStd
        FROM flights f
        JOIN carriers c ON f.CarrierCode = c.CarrierCode
        GROUP BY c.CarrierName
        HAVING NumberOfFlights > 100
        ORDER BY DepartureDelayStd ASC
    """
    consistent_airlines = spark.sql(query)
    
    # Write the result to a CSV file with overwrite mode
    consistent_airlines.write.csv(task2_output, header=True, mode='overwrite')
    print(f"Task 2 output written to {task2_output}")

# ------------------------
# Task 3: Origin-Destination Pairs with the Highest Percentage of Canceled Flights
# ------------------------
def task3_canceled_routes(flights_df, airports_df):
    flights_df.createOrReplaceTempView("flights")
    airports_df.createOrReplaceTempView("airports")
    
    query = """
        SELECT a1.AirportName AS OriginAirport, a1.City AS OriginCity,
               a2.AirportName AS DestinationAirport, a2.City AS DestinationCity,
               (SUM(CASE WHEN f.ActualDeparture IS NULL THEN 1 ELSE 0 END) / COUNT(f.FlightNum)) * 100 AS CancellationRate
        FROM flights f
        JOIN airports a1 ON f.Origin = a1.AirportCode
        JOIN airports a2 ON f.Destination = a2.AirportCode
        GROUP BY a1.AirportName, a1.City, a2.AirportName, a2.City
        ORDER BY CancellationRate DESC
    """
    canceled_routes = spark.sql(query)
    
    # Write the result to a CSV file with overwrite mode
    canceled_routes.write.csv(task3_output, header=True, mode='overwrite')
    print(f"Task 3 output written to {task3_output}")

# ------------------------
# Task 4: Carrier Performance Based on Time of Day
# ------------------------
def task4_carrier_performance_time_of_day(flights_df, carriers_df):
    flights_df.createOrReplaceTempView("flights")
    carriers_df.createOrReplaceTempView("carriers")
    
    query = """
        SELECT c.CarrierName,
               CASE
                   WHEN HOUR(f.ScheduledDeparture) BETWEEN 6 AND 11 THEN 'Morning'
                   WHEN HOUR(f.ScheduledDeparture) BETWEEN 12 AND 17 THEN 'Afternoon'
                   WHEN HOUR(f.ScheduledDeparture) BETWEEN 18 AND 23 THEN 'Evening'
                   ELSE 'Night'
               END AS TimeOfDay,
               AVG(unix_timestamp(f.ActualDeparture) - unix_timestamp(f.ScheduledDeparture)) / 60 AS AverageDepartureDelay
        FROM flights f
        JOIN carriers c ON f.CarrierCode = c.CarrierCode
        GROUP BY c.CarrierName, TimeOfDay
    """
    carrier_performance_time_of_day = spark.sql(query)

    # Adding ranking within each time of day
    window_spec_time = Window.partitionBy("TimeOfDay").orderBy("AverageDepartureDelay")
    carrier_performance_time_of_day = carrier_performance_time_of_day.withColumn("Rank", row_number().over(window_spec_time))
    
    # Write the result to a CSV file with overwrite mode
    carrier_performance_time_of_day.write.csv(task4_output, header=True, mode='overwrite')
    print(f"Task 4 output written to {task4_output}")

# ------------------------
# Call the functions for each task
# ------------------------
task1_largest_discrepancy(flights_df, carriers_df)
task2_consistent_airlines(flights_df, carriers_df)
task3_canceled_routes(flights_df, airports_df)
task4_carrier_performance_time_of_day(flights_df, carriers_df)

# Stop the Spark session
spark.stop()
