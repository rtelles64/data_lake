import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, to_timestamp
from pyspark.sql.functions import (year, month, dayofmonth,
                                   hour, weekofyear, date_format)

from pyspark.sql.types import TimestampType


config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID'] = config['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY'] = config['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    '''
    Creates Spark session

    Returns
    -------
    SparkSession
        A SparkSession object
    '''
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    '''
    Processes song data

    Parameters
    ----------
    spark : SparkSession object
        The Spark session to load/save data
    input_data : str
        Input file directory
    output_data : str
        Output parquet file directory
    '''
    # get filepath to song data file
    song_data = input_data + "song_data/*/*/*"

    # read song data file
    df = spark.read.json(song_data)

    # Creating a temp view to isolate info
    df.createOrReplaceTempView("songs_table_data")

    # extract columns to create songs table
    songs_table = spark.sql("""
        SELECT DISTINCT song_id, title, artist_id, year, duration
        FROM songs_table_data
    """)

    # write songs table to parquet files partitioned by year and artist
    songs_table.write.partitionBy("year", "artist_id") \
                     .parquet(output_data + "songs_table")

    # extract columns to create artists table
    artists_table = spark.sql("""
        SELECT DISTINCT artist_id, artist_name, artist_location,
                        artist_latitude, artist_longitude
        FROM songs_table_data
    """)

    # write artists table to parquet files
    artists_table.write.mode('overwrite') \
                       .parquet(output_data + "artists_table")


def process_log_data(spark, input_data, output_data):
    '''
    Processes log data

    Parameters
    ----------
    spark : SparkSession object
        The Spark session to load/save data
    input_data : str
        Input file directory
    output_data : str
        Output parquet file directory
    '''
    # get filepath to log data file
    log_data = input_data + "log_data/*/*/*"

    # read log data file
    df = spark.read.json(log_data)

    # filter by actions for song plays
    df = df.where(df.page == "NextSong")

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: to_timestamp(x))
    df = df.withColumn('timestamp', get_timestamp('ts'))

    # create datetime column from original timestamp column
    get_datetime = udf(lambda x: datetime.fromtimestamp(x/1000),
                       TimestampType())
    df = df.withColumn('date_time', get_datetime('ts'))

    # Create a view for easier read
    df.createOrReplaceTempView("log_data")

    # extract columns for users table
    users_table = spark.sql("""
        SELECT DISTINCT qry.userId, qry.firstName, qry.lastName,
                        qry.gender, qry.level
        FROM (
            SELECT date_time, userId, firstName, lastName, gender, level,
                   RANK() OVER (
                       PARTITION BY userId ORDER BY date_time DESC
                   ) AS rank
            FROM log_data
        ) AS qry
        WHERE qry.rank = 1
    """)

    # write users table to parquet files
    users_table.write.mode('overwrite').parquet(output_data + "users_table")

    # extract columns to create time table
    time_table = df.select(
        "date_time",
        hour("date_time").alias('hour'),
        dayofmonth("date_time").alias('day'),
        weekofyear("date_time").alias('week'),
        month("date_time").alias('month'),
        year("date_time").alias('year'),
        date_format("date_time", "u").alias('weekday')
    ).distinct()

    # write time table to parquet files partitioned by year and month
    time_table.write.partitionBy("year", "month") \
                    .parquet(output_data + "time_table")

    # read in song data to use for songplays table
    song_df = spark.read.parquet(output_data + "songs_table")

    # Create view for workability
    song_df.createOrReplaceTempView("songs_table")

    # extract columns from joined song and log datasets to create
    # songplays table
    songplays_table = spark.sql("""
        SELECT logs.datetime, times.year, times.month, logs.userId, logs.level,
               que.song_id, que.artist_id, logs.sessionId, logs.location,
               logs.userAgent
        FROM log_data logs
        JOIN time_table times ON logs.datetime = times.datetime
        LEFT JOIN (
            SELECT songs.song_id, songs.title, art.artist_id, art.artist_name
            FROM songs_table songs
            JOIN artists_table art ON songs.artist_id = art.artist_id
        ) AS que ON logs.song = que.title AND logs.artist = que.artist_name
    """)

    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.partitionBy("year", "month") \
                         .parquet(output_data + "songplays_table")


def main():
    '''
    The main function
    '''
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "s3a://data_eng/project4/"

    process_song_data(spark, input_data, output_data)
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
